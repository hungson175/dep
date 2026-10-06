"""The learn/deepseek-flash.ipynb recipe, without notebook or dotenv side effects.

One token, thinking off, temperature 1, top-20 logprobs. Candidate probabilities
are conditional on the supplied options; they are not empirically calibrated.
Missing candidates fail closed. No retries, logit bias, or free-text fallback.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class DistributionError(ValueError):
    """The reported token distribution cannot support this decision."""


class APIError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"DeepSeek HTTP {status}")


def marks(n):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 35:
        raise ValueError("options must contain 1 to 35 entries")
    return ([str(i) for i in range(1, 10)] + [chr(65 + i) for i in range(26)])[:n]


def confidence(p):
    top = sorted(p.values(), reverse=True)[:2]
    return round(top[0] - (top[1] if len(top) > 1 else 0.0), 4)


def build_request(state, question, model="deepseek-flash"):
    """Preserve the notebook's prompts and insertion order, including score 1..N."""
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("nonempty instructions required")
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    qtype, criteria = question.get("type"), question.get("criteria")
    prefix = f"Text:\n{state}\n\nQuestion: {instructions}"
    if qtype == "noul":
        crit = ""
        if criteria:
            if not isinstance(criteria, dict) or not {"true", "false"} <= criteria.keys():
                raise ValueError("noul criteria must include true and false")
            crit = f"\ntrue means: {criteria['true']}\nfalse means: {criteria['false']}"
        body = prefix + crit + "\nReply with exactly one word: Yes or No."
        labels = {"true": "Yes", "false": "No"}
    elif qtype in ("choice", "score"):
        if qtype == "choice" and isinstance(criteria, list):
            if any(not isinstance(k, str) for k in criteria) or len(set(criteria)) != len(criteria):
                raise ValueError("choice labels must be unique strings")
            criteria = {k: k for k in criteria}
        if qtype == "choice":
            if not isinstance(criteria, dict) or any(not isinstance(k, str) for k in criteria):
                raise ValueError("choice criteria must be a mapping or list of labels")
            keys = list(criteria)
            tokens = marks(len(keys))
            menu = "\n".join(f"{m}. {k} -- {criteria[k]}" for m, k in zip(tokens, keys))
            labels = dict(zip(keys, tokens))
            section = "Options"
        else:
            if not isinstance(criteria, list):
                raise ValueError("score criteria must be a list")
            tokens = marks(len(criteria))
            menu = "\n".join(f"{m}. {d}" for m, d in zip(tokens, criteria))
            labels = {str(i + 1): m for i, m in enumerate(tokens)}
            section = "Scale"
        body = (prefix + f"\n{section}:\n{menu}\n"
                f"Reply with exactly one character: {', '.join(tokens)}.")
    else:
        raise ValueError("question type must be noul, choice, or score")
    return {
        "model": model, "messages": [{"role": "user", "content": body}],
        "max_tokens": 1, "temperature": 1.0, "thinking": {"type": "disabled"},
        "logprobs": True, "top_logprobs": 20,
    }, labels


def candidate_probs(report, labels):
    """Sum case/space variants and normalize in log space, avoiding underflow."""
    norm = lambda s: s.strip().casefold()
    if not labels or len({norm(t) for t in labels.values()}) != len(labels):
        raise DistributionError("empty or ambiguous candidate marks")
    if not isinstance(report, list):
        raise DistributionError("top_logprobs must be a list")
    logs = {name: [] for name in labels}
    seen = set()
    for row in report:
        if not isinstance(row, dict):
            raise DistributionError("malformed top_logprobs row")
        token, lp = row.get("token"), row.get("logprob")
        if (not isinstance(token, str) or isinstance(lp, bool)
                or not isinstance(lp, (int, float)) or not math.isfinite(lp) or lp > 0):
            raise DistributionError("invalid token or logprob")
        if token in seen:
            raise DistributionError("duplicate token in top_logprobs")
        seen.add(token)
        for name, mark in labels.items():
            if norm(token) == norm(mark):
                if lp <= -9999:
                    raise DistributionError("candidate has a truncated logprob sentinel")
                logs[name].append(lp)
    missing = [name for name, values in logs.items() if not values]
    if missing:
        raise DistributionError(f"candidates missing from top-20 report: {missing}")
    peak = max(lp for values in logs.values() for lp in values)
    raw = {name: sum(math.exp(lp - peak) for lp in values) for name, values in logs.items()}
    z = sum(raw.values())
    mass = math.exp(peak) * z
    if mass > 1.000001:
        raise DistributionError("candidate mass exceeds one")
    return {k: v / z for k, v in raw.items()}, mass


def decode_response(out, labels):
    try:
        positions = out["choices"][0]["logprobs"]["content"]
        if not isinstance(positions, list) or len(positions) != 1:
            raise DistributionError("expected exactly one generated token position")
        report = positions[0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        raise DistributionError("missing first-token logprobs") from None
    return candidate_probs(report, labels)


@dataclass
class DeepSeekFlashClient:
    key: str = field(repr=False)
    model: str = "deepseek-flash"
    endpoint: str = "https://api.deepseek.com"
    timeout_s: float = 120.0

    def __post_init__(self):
        if not self.key or not self.key.strip():
            raise ValueError("DEEPSEEK_API_KEY must be exported")
        self.endpoint = self.endpoint.rstrip("/")
        if self.endpoint not in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
            raise ValueError("credentials may only be sent to the official DeepSeek endpoint")

    @classmethod
    def from_env(cls):
        # Intentionally no automatic .env loading or model-list request.
        return cls(os.environ.get("DEEPSEEK_API_KEY", ""))

    def post(self, body):
        req = urllib.request.Request(
            self.endpoint + "/chat/completions", json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Do not expose response bodies: providers can echo private input.
            raise APIError(e.code) from None

    def decide(self, state, question):
        body, labels = build_request(state, question, self.model)
        return decode_response(self.post(body), labels)[0]

    def noul(self, state, question, criteria=None):
        p = self.decide(state, {"type": "noul", "instructions": question, "criteria": criteria})
        return p["true"], confidence(p)

    def choice(self, state, question, options):
        p = self.decide(state, {"type": "choice", "instructions": question, "criteria": options})
        return p, confidence(p)

    def score(self, state, question, levels):
        p = self.decide(state, {"type": "score", "instructions": question, "criteria": levels})
        return sum(int(k) * v for k, v in p.items()), p, confidence(p)


@dataclass
class BenchmarkResult:
    ok: bool = False
    probs: dict | None = None
    probs_source: str = "native_logprobs_candidate_renormalized"
    model: str = "deepseek-flash"
    status: int | None = None
    error: str | None = None
    error_kind: str | None = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    raw: dict | None = None
    request_body: dict | None = None
    candidate_mass: float | None = None


class JevBenchAdapter:
    """Map notebook true/false and 1-based score to canonical JevBench keys."""
    name = "deepseek_flash_notebook"

    def __init__(self, client):
        self.client = client

    def prepare(self, task):
        # Read only state and question; never expected, rationale or provenance.
        body, labels = build_request(task.state, task.question, self.client.model)
        qtype = task.question["type"]
        rename = {k: k for k in labels}
        if qtype == "noul":
            rename = {"true": "yes", "false": "no"}
        elif qtype == "score":
            rename = {k: str(int(k) - 1) for k in labels}
        if set(rename.values()) != set(task.labels):
            raise ValueError("canonical task labels do not match the question criteria")
        return body, labels, rename

    def run(self, task):
        r = BenchmarkResult(model=self.client.model)
        try:
            body, labels, rename = self.prepare(task)
        except (ValueError, TypeError):
            r.error_kind, r.error = "input", "invalid task specification"
            return r
        r.request_body = body
        t0 = time.perf_counter()
        try:
            r.raw = self.client.post(body)
        except Exception as e:
            r.error_kind, r.error = "transport", type(e).__name__
            r.status = e.status if isinstance(e, APIError) else None
            return r
        finally:
            r.latency_s = time.perf_counter() - t0
        r.status = 200
        if isinstance(r.raw, dict):
            r.model = r.raw.get("model", self.client.model)
            r.usage = dict(r.raw.get("usage") or {})
            r.usage["input_tokens"] = r.usage.get("prompt_tokens")
            r.usage["output_tokens"] = r.usage.get("completion_tokens")
        try:
            p, r.candidate_mass = decode_response(r.raw, labels)
        except DistributionError as e:
            r.error_kind, r.error = "distribution", str(e)
            return r
        r.probs, r.ok = {rename[k]: v for k, v in p.items()}, True
        return r
