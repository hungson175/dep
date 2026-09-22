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


# ---- the three Jev primitives, built on decide() -------------------------

_ASK = ("<|im_start|>user\n{body}<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>Answer:")


def noul(state, question, criteria=None):
    """Yes/no -> probability of true."""
    crit = f"\ntrue means: {criteria['true']}\nfalse means: {criteria['false']}" if criteria else ""
    body = (f"Text:\n{state}\n\nQuestion: {question}{crit}\n"
            "Reply with exactly one word: Yes or No.")
    p = decide(_ASK.format(body=body), {"true": " Yes", "false": " No"})
    return p["true"]


def choice(state, question, options):
    """options: {name: description}. Returns {name: probability}."""
    keys = list(options)
    menu = "\n".join(f"{i+1}. {k} -- {options[k]}" for i, k in enumerate(keys))
    body = (f"Text:\n{state}\n\nQuestion: {question}\nOptions:\n{menu}\n"
            "Reply with exactly one digit.")
    p = decide(_ASK.format(body=body), {k: f" {i+1}" for i, k in enumerate(keys)})
    return p


def score(state, question, levels):
    """levels: ordered list of descriptions. Returns probability-weighted value."""
    menu = "\n".join(f"{i+1}. {d}" for i, d in enumerate(levels))
    body = (f"Text:\n{state}\n\nQuestion: {question}\nScale:\n{menu}\n"
            "Reply with exactly one digit.")
    p = decide(_ASK.format(body=body), {str(i+1): f" {i+1}" for i in range(len(levels))})
    return sum(int(k) * v for k, v in p.items())


# ---- schema -> JSON ------------------------------------------------------

def evaluate(state, schema, workers=4):
    """schema: {field: ("noul"|"choice"|"score", question, spec)} -> plain dict.

    Nothing is parsed: each answer is a token id looked up in the candidate map
    we built ourselves, so a malformed or out-of-schema value cannot occur.
    """
    from concurrent.futures import ThreadPoolExecutor
    kinds = {"noul": noul, "choice": choice, "score": score}

    def one(item):
        field, (kind, question, spec) = item
        fn = kinds[kind]
        return field, (fn(state, question) if spec is None
                       else fn(state, question, spec))

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, schema.items()))
