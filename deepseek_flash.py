"""The learn/deepseek-flash.ipynb recipe, without notebook or dotenv side effects.

One token, thinking off, temperature 1, top-20 logprobs. Candidate probabilities
are conditional on the supplied options; they are not empirically calibrated.
Missing candidates default to zero (a censored-probability approximation); use
missing_policy="error" for the original strict recipe. No retries or text fallback.
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


class TransportError(RuntimeError):
    """An API request failed before a usable JSON response was received."""


def validate_policy(policy):
    if policy not in ("zero", "error"):
        raise ValueError('missing_policy must be "zero" or "error"')


def marks(n):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 35:
        raise ValueError("options must contain 1 to 35 entries")
    return ([str(i) for i in range(1, 10)] + [chr(65 + i) for i in range(26)])[:n]


def confidence(p):
    top = sorted(p.values(), reverse=True)[:2]
    return round(top[0] - (top[1] if len(top) > 1 else 0.0), 4)


def build_request(state, question, model="deepseek-flash"):
    """Preserve the notebook's prompts and insertion order, including score 1..N."""
    if not isinstance(question, dict):
        raise ValueError("question must be a dictionary")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("nonempty instructions required")
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, allow_nan=False)
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


def candidate_probs(report, labels, *, missing_policy="zero"):
    """Sum variants, fill omitted options with zero, and normalize in log space.

    Zero represents an unreported option, not a measured zero probability.
    At least one candidate must be reported; malformed reports still fail.
    """
    validate_policy(missing_policy)
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
    if missing and missing_policy == "error":
        raise DistributionError(f"candidates missing from top-20 report: {missing}")
    if len(missing) == len(labels):
        raise DistributionError("no candidate tokens in top-20 report")
    peak = max(lp for values in logs.values() for lp in values)
    raw = {name: sum(math.exp(lp - peak) for lp in values) for name, values in logs.items()}
    z = sum(raw.values())
    mass = math.exp(peak) * z
    if mass > 1.000001:
        raise DistributionError("candidate mass exceeds one")
    return {k: v / z for k, v in raw.items()}, mass


def decode_response(out, labels, *, missing_policy="zero"):
    try:
        positions = out["choices"][0]["logprobs"]["content"]
        if not isinstance(positions, list) or len(positions) != 1:
            raise DistributionError("expected exactly one generated token position")
        report = positions[0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        raise DistributionError("missing first-token logprobs") from None
    return candidate_probs(report, labels, missing_policy=missing_policy)


def missing_options(out, labels):
    """Call only after decode_response has validated the report."""
    reported = {r["token"].strip().casefold()
                for r in out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
    return [name for name, mark in labels.items() if mark.strip().casefold() not in reported]


@dataclass
class Decision:
    probabilities: dict
    candidate_mass: float
    missing_options: list
    model: str
    usage: dict


@dataclass
class DeepSeekFlashClient:
    key: str = field(repr=False)
    model: str = "deepseek-flash"
    endpoint: str = "https://api.deepseek.com"
    timeout_s: float = 120.0
    missing_policy: str = "zero"

    def __post_init__(self):
        validate_policy(self.missing_policy)
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("DEEPSEEK_API_KEY must be exported")
        self.key = self.key.strip()
        if (isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("timeout must be a finite positive number")
        self.endpoint = self.endpoint.rstrip("/")
        if self.endpoint not in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
            raise ValueError("credentials may only be sent to the official DeepSeek endpoint")

    @classmethod
    def from_env(cls, *, missing_policy="zero"):
        # Intentionally no automatic .env loading or model-list request.
        return cls(os.environ.get("DEEPSEEK_API_KEY", ""), missing_policy=missing_policy)

    def post(self, body):
        req = urllib.request.Request(
            self.endpoint + "/chat/completions", json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        # urllib follows some redirects; never forward credentials to their target.
        req.add_unredirected_header("Authorization", f"Bearer {self.key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # Do not expose response bodies: providers can echo private input.
            raise APIError(e.code) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise DistributionError("API response is not valid JSON") from None
        except OSError as e:
            raise TransportError(type(e).__name__) from None

    def evaluate(self, state, question):
        """Return a typed decision plus diagnostics, with exactly one API call."""
        body, labels = build_request(state, question, self.model)
        out = self.post(body)
        p, mass = decode_response(out, labels, missing_policy=self.missing_policy)
        usage = out.get("usage")
        if usage is None:
            usage = {}
        if not isinstance(usage, dict):
            raise DistributionError("malformed API usage")
        return Decision(p, mass, missing_options(out, labels), out.get("model", self.model), usage)

    def decide(self, state, question):
        return self.evaluate(state, question).probabilities

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
    missing_options: list = field(default_factory=list)
    missing_policy: str = "zero"


class JevBenchAdapter:
    """Map notebook true/false and 1-based score to canonical JevBench keys."""
    name = "deepseek_flash_notebook"

    def __init__(self, client):
        self.client = client

    @property
    def missing_policy(self):
        return self.client.missing_policy

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
        r = BenchmarkResult(model=self.client.model, missing_policy=self.missing_policy)
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
            p, r.candidate_mass = decode_response(r.raw, labels, missing_policy=self.missing_policy)
        except DistributionError as e:
            r.error_kind, r.error = "distribution", str(e)
            return r
        r.missing_options = [rename[k] for k in missing_options(r.raw, labels)]
        r.probs, r.ok = {rename[k]: v for k, v in p.items()}, True
        return r
