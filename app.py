"""dep — typed-decision playground on top of a local llama.cpp server.

The backend serves 8 slots, so the queue admits 8 *model calls* -- not 8 requests.
A request with 4 questions fans out to 4 calls that run in parallel and compete for
the same 8 slots as everybody else's. Every response reports the critical path:
which branch decided the latency, how long it waited, and how long it was served.
"""
import asyncio, collections, hashlib, hmac, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import minijev

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = int(os.environ.get("DEP_SLOTS", "8"))     # must match llama-server -np
MAX_QUESTIONS = 4
MAX_WAITING = 96                                   # model calls queued before shedding
SLOT_TIMEOUT_S = 90
MAX_STATE_CHARS = 40_000
RATE_PER_MIN = 30

# Stress test is gated: without DEP_STRESS_HASH (sha256 hex of the password) the
# endpoint does not exist. The hash is never committed -- it lives in the unit file.
STRESS_HASH = os.environ.get("DEP_STRESS_HASH", "").strip().lower()
STRESS_N = 20
STRESS_COOLDOWN_S = 20
_stress_last = {"t": 0.0}

app = FastAPI(title="dep", docs_url="/api/docs", openapi_url="/api/openapi.json")

_slots = threading.Semaphore(SLOTS)   # one permit == one llama.cpp slot
_pool = ThreadPoolExecutor(max_workers=SLOTS * MAX_QUESTIONS, thread_name_prefix="dep")
_hits: Dict[str, collections.deque] = collections.defaultdict(collections.deque)
_lock = threading.Lock()

stats = {"active": 0, "waiting": 0, "calls": 0, "requests": 0, "rejected": 0,
         "peak_wait_ms": 0.0, "total_wait_ms": 0.0}


def _bump(field: str, by: int = 1):
    with _lock:
        stats[field] += by


def rate_limit(req: Request):
    ip = req.headers.get("cf-connecting-ip") or (req.client.host if req.client else "?")
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        wait = int(60 - (now - q[0])) + 1
        raise HTTPException(429, f"You have used this playground {RATE_PER_MIN} times "
                                 f"in the last minute, which is the per-IP limit. "
                                 f"Try again in about {wait}s.")
    q.append(now)


class Question(BaseModel):
    type: Literal["noul", "choice"]
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: Optional[Dict[str, str]] = None


class Req(BaseModel):
    state: str = Field(max_length=MAX_STATE_CHARS)
    questions: Dict[str, Question]


def _answer(state: str, name: str, q: Question) -> Dict[str, Any]:
    if q.type == "noul":
        crit = q.criteria if isinstance(q.criteria, dict) and q.criteria else None
        if crit and not {"true", "false"} <= set(crit):
            raise ValueError(f"{name}: noul criteria needs 'true' and 'false' keys")
        p, cert = minijev.noul(state, q.instructions, crit)
        return {"type": "noul", "noul": round(p, 4), "certainty": cert}
    if q.type == "choice":
        if not isinstance(q.criteria, dict) or len(q.criteria) < 2:
            raise ValueError(f"{name}: choice needs a criteria object with >=2 options")
        p, cert = minijev.choice(state, q.instructions, q.criteria)
        best = max(p, key=p.get)
    return {"type": "choice", "choice": best,
            "probabilities": {k: round(v, 4) for k, v in p.items()},
            "certainty": cert}


def _one_call(state: str, name: str, q: Question, t_arrive: float) -> Tuple[str, Dict, float, float]:
    """Runs in a pool thread. Blocks on a slot permit, then hits the model."""
    _bump("waiting")
    got = _slots.acquire(timeout=SLOT_TIMEOUT_S)
    _bump("waiting", -1)
    if not got:
        raise TimeoutError(f"Question '{name}' waited {SLOT_TIMEOUT_S}s and never got one "
                           f"of the {SLOTS} slots. The server is saturated right now.")
    t_slot = time.perf_counter()
    _bump("active")
    try:
        minijev.model_seconds_reset()
        ans = _answer(state, name, q)
        model_s = minijev.model_seconds()
    finally:
        _bump("active", -1)
        _bump("calls")
        _slots.release()
    return name, ans, (t_slot - t_arrive) * 1000, model_s * 1000


@app.post("/api/decide")
async def decide(body: Req, request: Request) -> Dict[str, Any]:
    t_arrive = time.perf_counter()
    rate_limit(request)
    n = len(body.questions)
    if not n:
        raise HTTPException(400, "at least one question required")
    if n > MAX_QUESTIONS:
        raise HTTPException(400, f"at most {MAX_QUESTIONS} questions per request")
    if stats["waiting"] >= MAX_WAITING:
        _bump("rejected")
        raise HTTPException(503, f"The queue is full: {stats['waiting']} model calls are "
                                 f"already waiting for {SLOTS} slots. Everyone is hammering "
                                 f"it at once — try again in a few seconds.")

    busy, waiting = stats["active"], stats["waiting"]
    loop = asyncio.get_running_loop()
    try:
        results = await asyncio.gather(*[
            loop.run_in_executor(_pool, _one_call, body.state, name, q, t_arrive)
            for name, q in body.questions.items()])
    except ValueError as e:
        raise HTTPException(400, str(e))
    except TimeoutError as e:
        _bump("rejected")
        raise HTTPException(504, str(e))
    except Exception as e:
        raise HTTPException(502, f"model backend error: {type(e).__name__}")

    total_ms = (time.perf_counter() - t_arrive) * 1000
    per_q = {name: {"queued": round(w, 1), "served": round(m, 1)}
             for name, _, w, m in results}
    # the branch that actually decided the latency
    crit_name, _, crit_wait, crit_model = max(results, key=lambda r: r[2] + r[3])
    routing = max(0.0, total_ms - crit_wait - crit_model)

    _bump("requests")
    with _lock:
        stats["total_wait_ms"] += crit_wait
        stats["peak_wait_ms"] = max(stats["peak_wait_ms"], crit_wait)

    return {
        "model": "bonsai-2-27b (ternary, local)",
        "answers": {name: ans for name, ans, _, _ in results},
        "cost_usd": 0.0,
        "timing_ms": {"queued": round(crit_wait, 1), "served": round(crit_model, 1),
                      "routing": round(routing, 1), "total": round(total_ms, 1)},
        "queue": {"slots": SLOTS, "model_calls": n, "ran": "parallel" if n > 1 else "single",
                  "critical_path": crit_name,
                  "busy_on_arrival": busy, "waiting_on_arrival": waiting,
                  "per_question_ms": per_q},
    }


class StressReq(BaseModel):
    password: str = Field(max_length=200)
    state: str = Field(max_length=MAX_STATE_CHARS)
    questions: Dict[str, Question]


@app.post("/api/stress")
async def stress(body: StressReq, request: Request) -> Dict[str, Any]:
    """Fire STRESS_N requests at once, server side. Password-gated: this is the one
    thing a visitor could use to saturate the box on purpose."""
    if not STRESS_HASH:
        raise HTTPException(404, "stress testing is not enabled on this instance")
    given = hashlib.sha256(body.password.encode()).hexdigest()
    if not hmac.compare_digest(given, STRESS_HASH):
        await asyncio.sleep(0.5)                      # blunt the guessing rate
        raise HTTPException(401, "wrong password")

    now = time.monotonic()
    left = STRESS_COOLDOWN_S - (now - _stress_last["t"])
    if left > 0:
        raise HTTPException(429, f"stress test cooling down, {left:.0f}s left")
    _stress_last["t"] = now

    n = len(body.questions)
    if not 1 <= n <= MAX_QUESTIONS:
        raise HTTPException(400, f"1..{MAX_QUESTIONS} questions")

    loop = asyncio.get_running_loop()

    async def one():
        t0 = time.perf_counter()
        res = await asyncio.gather(*[
            loop.run_in_executor(_pool, _one_call, body.state, name, q, t0)
            for name, q in body.questions.items()])
        total = (time.perf_counter() - t0) * 1000
        _, _, w, m = max(res, key=lambda r: r[2] + r[3])
        return {"queued": round(w, 1), "served": round(m, 1),
                "routing": round(max(0.0, total - w - m), 1), "total": round(total, 1)}

    t0 = time.perf_counter()
    rows = await asyncio.gather(*[one() for _ in range(STRESS_N)],
                                return_exceptions=True)
    wall = (time.perf_counter() - t0) * 1000
    ok = [r for r in rows if isinstance(r, dict)]
    ok.sort(key=lambda r: r["queued"])
    return {"requests": STRESS_N, "ok": len(ok), "slots": SLOTS,
            "calls": STRESS_N * n, "wall_ms": round(wall, 1), "rows": ok}


@app.get("/api/stats")
def get_stats() -> Dict[str, Any]:
    r = stats["requests"] or 1
    busy = stats["active"] >= SLOTS
    return {"slots": SLOTS, "active": stats["active"], "waiting": stats["waiting"],
            "requests": stats["requests"], "calls": stats["calls"],
            "rejected": stats["rejected"], "busy": busy,
            "max_waiting": MAX_WAITING, "max_questions": MAX_QUESTIONS,
            "rate_per_min": RATE_PER_MIN, "slot_timeout_s": SLOT_TIMEOUT_S,
            "stress_enabled": bool(STRESS_HASH), "stress_n": STRESS_N,
            "avg_queued_ms": round(stats["total_wait_ms"] / r, 1),
            "peak_queued_ms": round(stats["peak_wait_ms"], 1)}


@app.get("/api/health")
def health() -> Dict[str, Any]:
    try:
        minijev.token_id(" Yes")
        return {"ok": True, "slots": SLOTS, "active": stats["active"],
                "waiting": stats["waiting"], "max_questions": MAX_QUESTIONS}
    except Exception as e:
        raise HTTPException(503, f"backend down: {type(e).__name__}")


app.mount("/static", StaticFiles(directory=f"{HERE}/static"), name="static")


@app.get("/")
def index():
    return FileResponse(f"{HERE}/static/index.html")
