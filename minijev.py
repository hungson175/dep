"""mini-Jev: typed decisions from ANY llama.cpp server, one forced token + logit_bias.

Trick: generate exactly 1 token, add an equal large bias to every candidate token so
they dominate the distribution, then softmax the returned logprobs over the candidates
only. Equal bias cancels in the softmax ratio, so the renormalised distribution is the
model's true relative belief -- calibrated, not a guess parsed out of prose.
"""
import functools, json, math, os, threading, time, urllib.request

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
def token_id(text):
    """Single token id for `text`, trying the leading-space variant first.

    Which of " Yes" / "Yes" is one token is tokeniser-specific, so try both
    rather than hard-coding an assumption about the vocab.
    """
    for cand in ([text, text.lstrip()] if text.startswith(" ") else [text, " " + text]):
        ids = _post("/tokenize", {"content": cand})["tokens"]
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"no single-token form of {text!r}")


def decide(prompt, labels):
    """labels: {name: single-token string}. Returns {name: probability}, sums to 1."""
    ids = {name: token_id(tok) for name, tok in labels.items()}
    out = _post("/completion", {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0,
        "n_probs": max(20, len(ids) * 2),
        "logit_bias": [[i, BIAS] for i in ids.values()],
    })
    top = {t["id"]: t["logprob"]
           for t in out["completion_probabilities"][0]["top_logprobs"]}
    lp = {name: top.get(i, -60.0) for name, i in ids.items()}
    m = max(lp.values())
    e = {k: math.exp(v - m) for k, v in lp.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


# ---- confidence ----------------------------------------------------------


def confidence(p):
    """Top probability minus the runner-up.

    Not max(p): that is floored at 1/N, so 0.50 means "coin flip" with two options
    and "quite sure" with six, and it cannot tell {.50, .49} from {.50, .05 x10}.
    The margin is 0 for a dead tie and 1 for certainty whatever N is.
    """
    top2 = sorted(p.values(), reverse=True)[:2]
    return round(top2[0] - (top2[1] if len(top2) > 1 else 0.0), 4)


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
    menu = "\n".join(f"{i+1}. {k} -- {options[k]}" for i, k in enumerate(keys))
    body = (f"Text:\n{state}\n\nQuestion: {question}\nOptions:\n{menu}\n"
            "Reply with exactly one digit.")
    p = decide(_ASK.format(body=body), {k: f" {i+1}" for i, k in enumerate(keys)})
    return p, confidence(p)


def score(state, question, levels):
    """Ordered rubric -> probability-weighted value, plus the distribution.

    Same one token as choice; the only difference is the last line. choice takes
    the argmax because its options have no order. Here they do, so the expectation
    E[i] = sum(i * p_i) is meaningful and turns N integer levels into one real
    number -- no fine-tuning, no regression head.
    """
    menu = "\n".join(f"{i+1}. {d}" for i, d in enumerate(levels))
    body = (f"Text:\n{state}\n\nQuestion: {question}\nScale:\n{menu}\n"
            "Reply with exactly one digit.")
    p = decide(_ASK.format(body=body), {str(i+1): f" {i+1}" for i in range(len(levels))})
    return sum(int(k) * v for k, v in p.items()), p, confidence(p)
