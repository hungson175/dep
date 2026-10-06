# Bonsai-Dep — public JevBench diagnostic, 2026-10-06

## Result

**187/231 correct (80.95%)**, with **231/231 valid distributions (100%)**.
One serial pass completed on the existing local Bonsai server: exactly 231
one-token completion requests and 14 cached-mark tokenization requests. There
were no retries, missing candidates, transport failures, or distribution failures.
All decisions, including the 44 valid-but-wrong answers, remain in the denominator.

This is the same **231 legacy public authored cases** used for our DeepSeek run,
not the current public/sealed leaderboard pool. **No official score or rank**
is claimed, and no external benchmark submission was made.

| Request type | Correct / planned | Accuracy |
|---|---:|---:|
| choice | 117/139 | 84.17% |
| noul | 56/74 | 75.68% |
| score | 14/18 | 77.78% |

| Authored dataset | Correct / planned | Accuracy |
|---|---:|---:|
| easy | 48/48 | 100.00% |
| original | 67/72 | 93.06% |
| hard | 72/111 | 64.86% |

The weakest families were temporal/numeric (4/15), tradeoffs (2/6), probability
(5/10), and long policy (11/19). Simple intent, extraction, routing and tool
selection each scored 100% on these small public subsets; this is not evidence
of universal reliability or prompt-injection immunity.

## Recipe and runtime

- Model alias verified by the server's `/v1/models` metadata: `bonsai-2-27b`.
- Observed server model file: `Ternary-Bonsai-2-27B-PQ2_0.gguf`.
- Observed GPU: NVIDIA GeForce RTX 3090 Ti; the server remained shared with
  the playground, with 8 configured slots and total configured context 131072.
- Existing `minijev.py` prompts, answer marks, equal +50 logit bias,
  temperature 1, one-token strategy and probability normalization were unchanged.
- `post_sampling_probs=True`, `n_probs=min(40,max(20,2*N))`; truncating samplers
  stayed disabled. Missing native candidates still fail strictly.
- Adapter maps `true/false` to JevBench `yes/no`, and 1-based score labels to
  0-based indices. It never sends gold answers or task provenance to the model.
- Only the eager top-level `_load_dotenv()` call was removed from the in-memory
  AST during loading. The source file was not modified; `.env` was not read.
- Only the authorized local server key file was runtime-loaded. Keys and
  authorization headers were not printed or saved in run evidence.

## Latency, probability diagnostics and cost

| Metric | Result |
|---|---:|
| HTTP round-trip p50 | 399.65 ms |
| HTTP round-trip p95 | 2209.61 ms |
| Sum of measured decision round trips | 175.20 s |
| Top-label ECE, 10 equal-width bins | 0.03285 |
| Multiclass Brier mean | 0.24794 |
| Ordinal expected-value MAE, 18 score decisions | 0.29485 |

Latency includes initial uncached tokenization and the HTTP/model path, but not
an exclusive-GPU load test. Longer hard prompts account for much of the tail;
these numbers are **not throughput measurements** or comparable to a warmed
encoder-only kernel timing.

All 231 native candidate reports were complete. Their reported candidate mass
ranged from 0.9999999627 to 1.0000000298 (floating-point noise), before the
original candidate-only normalization. No zero-fill or verbalized confidence
was introduced. ECE uses the **maximum candidate probability**, not the
playground's top-minus-runner-up confidence margin. These are public-set
calibration diagnostics, not fitted or independently validated calibration.

**Provider API dollar spend: $0.** Local hardware/electricity cost was not
measured, so inference cost and price per 1000 decisions are **unknown**, not
invented as zero. No service restart, deployment, dependency installation,
model download, Git push or API-provider call occurred for this run.

## Same-public-item comparisons

| Recipe | Correct / 231 | Accuracy | Evidence status |
|---|---:|---:|---|
| Bonsai-Dep, current local recipe | 187 | 80.95% | Fresh serial local run |
| DeepSeek notebook, original strict decoding | 178 | 77.06% | Earlier fresh API run |
| DeepSeek notebook, zero-fill decoding | 190 | 82.25% | Offline replay of the same API responses |
| Jev 1.13.0 | 200 | 86.58% | Historical upstream records, not rerun |
| DeepSeek V4.1 Flash, thinking default | 226 | 97.84% | Historical upstream records, different recipe, not rerun |

Bonsai is 9 decisions above our strict DeepSeek decoding and 3 below our new
zero-fill decoding. Against zero-fill, Bonsai alone gets 8 cases right and
DeepSeek alone gets 11 right. These small differences from one public-set pass
do not establish general superiority. Models, hosts and decoding recipes differ;
historical comparisons are not fresh controlled reruns or current ranking scores.

Our DeepSeek [strict report](deepseek_flash_public_20261006.md) and
[zero-fill report](deepseek_flash_zero_fill_replay_20261006.md) are unchanged.
Historical baselines come from the pinned upstream
[per-task record](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/results/v1.2/jevbench-v1.2-per-task.json).

## Verification and evidence

- Pinned MIT-licensed upstream source:
  [`bb05a335bc809e61b20c0f745d25499a82b326fc`](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc).
  Existing cache verification checks the source manifest and Git blob hashes.
- 13 mocked/offline tests passed before inference; **99% statement coverage**
  for the new adapter/runner (204 statements; no LLM calls in tests).
- A post-run offline audit guarded all networking and replayed all 231 decisions:
  all 245 HTTP request bodies and all normalized distributions matched exactly.
  No additional inference or credential reads were needed for the audit.
- Durable manifest retains task IDs, dataset hash, source hashes, settings and
  metadata identity; records retain task hashes, scoring and latency.
- Raw evidence contains public request bodies and server responses, never headers.
  It remains git-ignored under
  `benchmark_runs/bonsai_dep_public_20261006T155052+0700/`.
- Evidence files: `manifest.json`, `runtime_snapshot.json`, `records.jsonl`,
  `summary.json`, `offline_replay_audit.json`, and 231 `raw/*.json` files.
- Records SHA-256:
  `0d33527e58544fa9b5aaa6a5583d270765a0bfe6186e15e5431e4d7d93491c19`.
- Summary SHA-256:
  `073e876a7f63942c5623b0d4ed2d54dce962dee401856895bbf49a5c73dba561`.

Reproduction is an explicit new run, not an automatic retry:

```bash
# Requires the existing pinned public cache and authorized local server.
python3 -m unittest discover -s tests -p 'test_jevbench_bonsai.py' -v
python3 -m benchmarks.jevbench_bonsai dry-run  # no key read or inference
python3 -m benchmarks.jevbench_bonsai run     # serial; new evidence folder
```

## Bottom summary

- Status: one public-only Bonsai-Dep run complete; no retry or external submission.
- Result: 187/231 correct (80.95%); all 231 native distributions valid.
- Types: choice 117/139, noul 56/74, score 14/18.
- Latency: p50 400 ms, p95 2210 ms on a shared RTX 3090 Ti.
- Cost: no provider API billing; hardware/electricity cost unmeasured.
- Comparison: above our strict DeepSeek, below zero-fill and historical Jev.
- Need from Boss: nothing for this run; official submission is a separate action.
