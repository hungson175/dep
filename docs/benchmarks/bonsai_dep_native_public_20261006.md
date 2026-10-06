# Bonsai-Dep - native local public diagnostic, 2026-10-06

## Correction and result

The earlier 400/2210 ms numbers are HTTP round trips, **not native inference latency**.
They remain archived in [the original HTTP report](bonsai_dep_public_20261006.md).
This fresh run executes libllama directly in a C++ process on the machine: no HTTP,
web server, API key, web queue or provider API call in the measured path.

**188/231 correct (81.39%); 231/231 valid distributions.**
**Native decision wall time: p50 139.670 ms; p95 1724.344 ms.**
The wall timer includes native input tokenization, KV reset, synchronized prefill,
first-answer candidate probabilities and small native orchestration overhead.
It excludes model loading, context initialization, warmups and file I/O.
Prompt strings are prepared before the native invocation; this is not a web/API
end-to-end latency or a throughput claim.

| Type | Correct/planned | Raw accuracy |
|---|---:|---:|
| choice | 118/139 | 84.89% |
| noul | 56/74 | 75.68% |
| score | 14/18 | 77.78% |

## Native timing breakdown

| Component | p50 ms | p95 ms |
|---|---:|---:|
| tokenize | 0.885 | 11.145 |
| reset | 0.546 | 0.670 |
| prefill | 112.954 | 1691.519 |
| sample | 22.726 | 24.046 |
| decision | 137.855 | 1714.218 |
| local_total | 139.467 | 1724.128 |
| local_wall | 139.670 | 1724.344 |

Component percentiles are computed separately and need not sum to the total
percentile. `decision` means prefill plus sampler only; `local_wall` is the
headline native call measurement. GPU synchronization brackets each prefill
batch. The first answer comes from the prompt's final logits: **no extra
answer-token forward pass** is added. Internal llama performance counters are
auxiliary because one-token prompt chunks can be classified as decode.

## Runtime and controls

- Exact GGUF: `Ternary-Bonsai-2-27B-PQ2_0.gguf`.
- GPU: NVIDIA GeForce RTX3090Ti (24 GB); CPU: Intel Core i9-13900K.
- Exact existing llama.cpp-Prism source/library revision:
  `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`.
- Explicit native CUDA backend loading; **65/65 layers offloaded**, verified
  through loader evidence. No silent CPU fallback.
- Context4096, batch2048, microbatch512, one sequence, 8 CPU threads,
  flash attention enabled, full GPU offload.
- One model load; two excluded warmups; 231 serial measured decisions.
- KV memory is cleared for each case. No cross-case prefix cache.
- Original minijev prompts, answer marks, equal+50 bias, temperature1,
  native float sampler and strict candidate-report policy are preserved.
- Existing Bonsai service stopped with Boss approval at16:46:43+07 and restored
  at16:48:43+07. A10-minute transient recovery timer guarded the stop and was
  removed after restoration. App and DeepSeek service configuration unchanged.
- Model metadata and public site health verified after restoration.
- No DeepSeek rerun. Provider spend caused by this native benchmark:$0.
  Local hardware/electricity cost remains **unknown**, not zero inference cost.

## Offline audit and differences

All231 prompt strings, candidate token IDs and prompt token counts matched the
archived HTTP run exactly. A separate no-network audit recomputed all scoring
and timing aggregates from saved native output, with no further inference.

This is a fresh numerical execution, not cached-response retiming. One prediction
changed (`hard-sol-c-multi_hop-12`, wrong to correct); maximum absolute
probability difference against the old execution was0.01311049.
The native single-sequence/context/batch execution differs from the shared
8-slot server; the new188 result must not be represented as the old187 run.
Brier=0.24756969; top-label ECE=0.02887708; ordinal
expected-value MAE=0.29491309. These are legacy diagnostic metrics.
Against DeepSeek zero-fill: Bonsai-only correct=8,
DeepSeek-only correct=10; no general winner claim.

## Current official protocol is NOT matched

The [current board](https://benchmarkheaven.com/jev-models) and
[official aggregate](https://benchmarkheaven.com/api/jevbench/v1.6.1), checked
2026-10-06, declare v1.6.1:300public+1200sealed, scorer setting O1S.
The public GitHub main revision checked is still our pinned legacy revision,
with `easy.jsonl`, `original.jsonl`, `hard.jsonl` (231 items). Its tree and
published releases do not provide the P300 artifact or `score_v16.py` sources.
We have **not obtained, hash-verified or executed that official harness**.

Published official scorer hash:
`02132271117653e6302c509232a581d0094f0e78eac8b1bfc67596fd5d5097e9`.
Published P manifest hash:
`6b2321f8359d87769a95c8e52135542d95ccdc353197e1e5c96abb9b616ce5ba`.

Raw accuracy is not their Intelligence. Their typed chance correction,
calibration, tier/type weighting, sealed-gap penalty, pricing and self-hosted
speed adjustment are not replaced by these diagnostics. **No official
Intelligence, Calibration axis, Speed, Cost, Capability, Composite or rank** is
computed. Applying the published2x+0.15s assumption to these latency numbers
alone would not make this a protocol-conformant score.

Next requirement: obtain the exact public P300 data/golds/manifest and official
scorer/dependencies, verify the published hashes and run conformance tests.
The sealed evaluation/ranking requires the benchmark organizer. We do not
invent a replacement public set, approximate scorer or sealed result.

## Evidence and tests

- Scope:231legacy authored public items; upstream pin
  `bb05a335bc809e61b20c0f745d25499a82b326fc`.
- Dataset SHA256: `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
- Model GGUF SHA256: `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
- Native output SHA256: `f04f383e89cbff8f8017bc21f771c6b70a9b8c467b1d70ae8ef18e9c96549c8d`.
- Scored records SHA256: `f36834f59798601715c9879d26147be80bc77ea3a3c43df739777ef45e93a272`.
- Evidence kept locally under `benchmark_runs/bonsai_dep_native_public_20261006T164643+0700`:
  native jobs/output, manifest, records/summary, binary, offline audit,
  loader log and service-restoration verification. No headers/keys included.
- 7 mocked Python seam tests passed before inference; the final suite has
  8 seam tests and100% combined line/branch coverage for the Python runner.
- 6 no-model C++ tests passed before inference. C++ unit line coverage49.44%
  (GPU/model/decode paths excluded); the real231-item run exercises those
  critical GPU seams. No combined coverage percentage is invented.

Explicit reproduction (a new inference run, never an automatic retry):

```bash
python3 -m benchmarks.jevbench_bonsai_native dry-run
# Build benchmarks/bonsai_native.cpp against the EXACT installed llama headers
# and libraries, then explicitly choose a new evidence directory.
python3 -m benchmarks.jevbench_bonsai_native run \
  --binary /path/to/bonsai-native --output benchmark_runs/NEW_DIRECTORY
```

## Bottom summary

- Native run complete:188/231(81.39%), all231valid; no retry.
- Native wall latency:p50 139.67ms,p95 1724.34ms.
- Engine prefill+sampler:p50 137.86ms,p95 1714.22ms.
- Bonsai service restored and healthy after2minutes; no DeepSeek rerun.
- Original HTTP evidence preserved; no official score/rank claimed.
- Need:verified v1.6.1 P300+O1S scorer artifacts for matching-metrics evaluation.
