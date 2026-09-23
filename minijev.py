"""mini-Jev: typed decisions from ANY llama.cpp server, one forced token + logit_bias.

Trick: generate exactly 1 token, add an equal large bias to every candidate token so
they dominate the distribution, then softmax the returned logprobs over the candidates
only. Equal bias cancels in the softmax ratio, so the renormalised distribution is the
model's true relative belief -- calibrated, not a guess parsed out of prose.
"""
import functools, json, os, threading, time, urllib.request

def _load_dotenv():
    """Read .env next to this file. Real environment variables win."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(p):
        return
    for line in open(p):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()

URL = os.environ.get("DEP_LLAMA_URL", "http://127.0.0.1:8080")
BIAS = 50.0


def _key():
    """API key for llama-server, if it was started with one. Optional."""
    k = os.environ.get("DEP_API_KEY")
    if k:
        return k.strip()
    path = os.environ.get("DEP_API_KEY_FILE")
    if path and os.path.exists(os.path.expanduser(path)):
        return open(os.path.expanduser(path)).read().strip()
    return ""


KEY = _key()


_local = threading.local()          # per-thread, so 8 concurrent requests do not mix


def model_seconds_reset():
    _local.t = 0.0


def model_seconds():
    return getattr(_local, "t", 0.0)


def _post(path, body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    finally:
        _local.t = getattr(_local, "t", 0.0) + (time.perf_counter() - t0)


@functools.lru_cache(maxsize=512)
def _token_id_cached(text):
    """Single token id for `text`, trying the leading-space variant first.

    Which of " Yes" / "Yes" is one token is tokeniser-specific, so try both
    rather than hard-coding an assumption about the vocab.
    """
    for cand in ([text, text.lstrip()] if text.startswith(" ") else [text, " " + text]):
        ids = _post("/tokenize", {"content": cand})["tokens"]
        if len(ids) == 1:
            return ids[0]
    raise ValueError("no single-token form for that answer mark")


def token_id(text):
    """Cache only short marks; caching arbitrary user strings pins unbounded RAM."""
    return _token_id_cached(text) if len(text) <= 8 else _token_id_uncached(text)


def _token_id_uncached(text):
    for cand in ([text, text.lstrip()] if text.startswith(" ") else [text, " " + text]):
        ids = _post("/tokenize", {"content": cand})["tokens"]
        if len(ids) == 1:
            return ids[0]
    raise ValueError("no single-token form for that answer mark")


def decide(prompt, labels):
    """labels: {name: single-token string}. Returns {name: probability}, sums to 1.

    Two things force the answer into the schema, and the prompt is neither of them:
      * n_predict=1  -- a hard stop, not a request. One token comes out, full stop.
      * logit_bias   -- an equal bias on every candidate, so the one token that does
                        come out must be one of ours.

    The bias must also be visible in the numbers we read back. llama.cpp's default
    `n_probs` reports PRE-sampling logits (server-context.cpp:1697), i.e. before
    logit_bias, ranked by the unbiased distribution -- so a candidate the model
    dislikes falls outside the reporting window and silently reads as zero.
    `post_sampling_probs` reports the distribution the sampler actually saw, where
    the biased candidates sit at the top. The other samplers truncate, so they are
    switched off, and temperature must be 1.0 or the reported distribution collapses
    to one-hot.
    """
    ids = {name: token_id(tok) for name, tok in labels.items()}
    out = _post("/completion", {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 1.0,
        "n_probs": min(40, max(20, len(ids) * 2)),
        "post_sampling_probs": True,
        "logit_bias": [[i, BIAS] for i in ids.values()],
        "top_k": 0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0, "top_n_sigma": -1.0,
    })
    pos = out["completion_probabilities"][0]
    top = {t["id"]: t["prob"] for t in pos.get("top_probs", [])}
    missing = [n for n, i in ids.items() if i not in top]
    if missing:   # would silently become zero -- refuse rather than invent a number
        raise RuntimeError(f"candidates missing from the reported distribution: {missing}")
    raw = {name: top[i] for name, i in ids.items()}
    z = sum(raw.values())
    if z <= 0:
        raise RuntimeError("all candidates had zero probability")
    return {k: v / z for k, v in raw.items()}


# ---- confidence ----------------------------------------------------------


def confidence(p):
    """Top probability minus the runner-up.

    Not max(p): that is floored at 1/N, so 0.50 means "coin flip" with two options
    and "quite sure" with six, and it cannot tell {.50, .49} from {.50, .05 x10}.
    The margin is 0 for a dead tie and 1 for certainty whatever N is.
    """
    top2 = sorted(p.values(), reverse=True)[:2]
    return round(top2[0] - (top2[1] if len(top2) > 1 else 0.0), 4)


def _marks(n):
    """n single-token answer marks. Digits run out at 9 (" 10" is two tokens),
    so anything larger continues with letters."""
    digits = [str(i) for i in range(1, 10)]
    letters = [chr(ord("A") + i) for i in range(26)]
    pool = digits + letters
    if n > len(pool):
        raise ValueError(f"at most {len(pool)} options, got {n}")
    return pool[:n]


# ---- the primitives, built on decide() ----------------------------------

_ASK = ("<|im_start|>user\n{body}<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>Answer:")


def noul(state, question, criteria=None):
    """Yes/no -> probability of true."""
    crit = f"\ntrue means: {criteria['true']}\nfalse means: {criteria['false']}" if criteria else ""
    body = (f"Text:\n{state}\n\nQuestion: {question}{crit}\n"
            "Reply with exactly one word: Yes or No.")
    p = decide(_ASK.format(body=body), {"true": " Yes", "false": " No"})
    return p["true"], confidence(p)


def choice(state, question, options):
    """options: {name: description}. Returns {name: probability}."""
    keys = list(options)
    marks = _marks(len(keys))
    menu = "\n".join(f"{m}. {k} -- {options[k]}" for m, k in zip(marks, keys))
    body = (f"Text:\n{state}\n\nQuestion: {question}\nOptions:\n{menu}\n"
            f"Reply with exactly one character: {', '.join(marks)}.")
    p = decide(_ASK.format(body=body), {k: f" {m}" for k, m in zip(keys, marks)})
    return p, confidence(p)


def score(state, question, levels):
    """Ordered rubric -> probability-weighted value, plus the distribution.

    Same one token as choice; the only difference is the last line. choice takes
    the argmax because its options have no order. Here they do, so the expectation
    E[i] = sum(i * p_i) is meaningful and turns N integer levels into one real
    number -- no fine-tuning, no regression head.
    """
    marks = _marks(len(levels))
    menu = "\n".join(f"{m}. {d}" for m, d in zip(marks, levels))
    body = (f"Text:\n{state}\n\nQuestion: {question}\nScale:\n{menu}\n"
            f"Reply with exactly one character: {', '.join(marks)}.")
    # rank i (1-based) carries the value, whatever mark was used to elicit it
    p = decide(_ASK.format(body=body), {str(i + 1): f" {m}" for i, m in enumerate(marks)})
    return sum(int(k) * v for k, v in p.items()), p, confidence(p)
