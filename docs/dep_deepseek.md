# Dep DeepSeek Python library

Ask typed questions and receive answers with a complete probability map. The
library handles prompts, one-token requests, option matching and normalization.
No model download or runtime dependencies are needed. Python 3.10+ is supported.

## Install from source

From this repository, in your own virtual environment:

```bash
python -m pip install .
```

This package is not yet published to PyPI. Existing users need the updated
source or the locally built wheel, not an unpushed GitHub URL.

To install the built wheel directly:

```bash
python -m pip install /path/to/dep_deepseek-0.1.0-py3-none-any.whl
```

Export your own `DEEPSEEK_API_KEY` securely in the launching environment.
Imports do not read `.env` files, contact a model, or expose keys.

## Choose an option

```python
from dep_deepseek import DeepSeek

dep = DeepSeek()  # Uses exported DEEPSEEK_API_KEY.

answer = dep.choice(
    "Please cancel my order.",
    "What does the customer want?",
    {"cancel": "Cancel an order", "track": "Track a shipment", "other": None},
)
print(answer["choice"])
print(answer["probabilities"])  # Every supplied option is present; values sum to 1.
```

Alternatively, pass a key held in memory: `DeepSeek(api_key=key)`.
Do not hardcode a real key in source or a notebook. `repr(dep)` redacts the key.

## Ask multiple typed questions

```python
result = dep.predict(
    {"message": "My payment failed again. Help me now!"},
    {
        "intent": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"payments": "Payment issues", "sales": "Buying advice", "other": None},
        },
        "urgent": {"type": "noul", "instructions": "The customer needs urgent help."},
        "mood": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["calm", "annoyed", "angry"],
        },
    },
)

print(result["answers"]["intent"]["choice"])
print(result["answers"]["urgent"]["noul"])
print(result["answers"]["mood"]["score"])
print(result["usage"])
```

One API request is made per question, serially. All questions are validated
before any request is sent. Invalid API responses and access/network failures
raise errors; completed calls can still be billed if a later question fails.
The library does not retry, cache, load credentials from disk, or impose a
spending cap. Budget and rate controls belong to the caller; this repository's
separate benchmark runner keeps its shared $20 ledger.

## Answers

| Type | Main value | Probability map |
|---|---|---|
| `choice` | `choice`: one supplied label | Exact supplied option keys |
| `noul` | `noul`: probability of true | `true`, `false` |
| `score` | `score`: expected **1-based** level | `"1"` through `"N"`; `legend` maps keys to descriptions |

Every answer includes `type`, `probabilities`, and `confidence`. Confidence is
the top probability minus the runner-up, rounded to four decimals. Ties pick
the first supplied option. Choice criteria can also be a list of unique labels.
State can be text or a JSON-serializable value; up to 35 options are supported.
Convenience methods `dep.noul(state, instructions, criteria=None)` and
`dep.score(state, instructions, levels)` return the same typed answer shapes.

## Missing options default to zero

If an option is absent from DeepSeek's reported first-token probabilities,
the library returns **0.0 for that option** and normalizes the available option
probabilities to sum to one. This is a deliberate fallback, **not proof that the
model assigned exactly zero probability**. The reported window is incomplete.

`answer["diagnostics"]` identifies `missing_options`, `missing_policy`, and the
`candidate_mass` observed before normalization. These diagnostics can be ignored
for ordinary routing, or inspected when evaluating reliability. Zero-filling
can inflate confidence; the probabilities are not empirically calibrated.

To require every option to be reported instead:

```python
strict = DeepSeek(missing_policy="error")
```

When no candidate is reported, both modes raise `DistributionError` instead
of inventing a uniform distribution. Malformed/non-finite probabilities still
raise errors in both modes. Catch `APIError` for HTTP errors, `TransportError`
for network failures, and `DistributionError` for unusable model responses.
Ordinary input validation raises `ValueError` or JSON serialization errors.

## Model recipe and limits

The fixed model is `deepseek-flash` (DeepSeek V4.1 Flash). It uses thinking
disabled, one output token, temperature 1, and top-20 logprobs. The implementation
uses the original notebook prompts; the new default changes missing-option
handling only. No logit bias, prose parsing, fine-tuning, or local GPU is used.

This is a hosted inference adapter, not a new trained model. Schema-conforming
outputs do not guarantee correct decisions or prompt-injection resistance.
Do not use it as the sole control for high-stakes decisions. Benchmarks are
public-only diagnostics, not an official leaderboard rank.

## Development checks

With Python's `build` and `coverage` development tools available:

```bash
python -m build --wheel --sdist
DEP_TEST_WHEEL="$PWD/dist/dep_deepseek-0.1.0-py3-none-any.whl" \
  python -m coverage run --branch --source=deepseek_flash,dep_deepseek,benchmarks \
  -m unittest discover -s tests -v
python -m coverage report -m
```

The wheel test installs into a temporary directory outside the repository and
checks a mocked request end-to-end. Tests do not need real credentials or API
calls; the fixture includes one sanitized recorded missing-option response.

## Bottom summary

- Import: `from dep_deepseek import DeepSeek`.
- Use: `predict`, `choice`, `noul`, or `score`; no model-specific plumbing.
- Default: omitted options return 0.0, with complete normalized probability maps.
- Optional strict mode: `missing_policy="error"`.
- Credentials and API costs belong to the consumer; no automatic retries or budget cap.
- Status: local package, not published; no action needed from Boss to use local source.
