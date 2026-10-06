# Original Jev: 231-case public diagnostic — 6 October 2026

## Result

**199/231 correct (86.15%)**, all 231 valid and strictly valid; no renormalization, no retries, no missing-option synthesis. The provider reported **jev-1.13.0** for every response. All calls returned HTTP 200.

| Type | Correct | Accuracy |
|---|---:|---:|
| choice | 124/139 | 89.21% |
| noul | 61/74 | 82.43% |
| score | 14/18 | 77.78% |

API wall round-trip p50 **418.09 ms**, p95 **516.34 ms** across all 231 requests. This includes request preparation, network and provider time, and adapter parsing; it is not a measurement of internal model compute. Calls were serial with 50 ms between calls; that inter-call delay is outside recorded latency. No warmup or extra paid probe.

Usage-derived charge: **$0.008874978** total; **$0.038419818 per 1,000 decisions** at the published $0.042/M input tariff, output free. This is not a provider invoice. A separate durable $20 cap/reservation ledger was used; every call was settled from returned usage. [TypeSafe model/pricing documentation](https://docs.typesafe.ai/models), checked 6 October 2026.

Legacy diagnostic Brier: 0.179995671; top-label ECE: 0.048095238; ordinal MAE: 0.132777778. These are not current official Intelligence/Calibration axes.

## Fairness and scope

Same 231 authored public cases and dataset hash as Dep's earlier runs. Same legacy label/ordinal expected-value scoring and denominator; failures would count wrong. Jev uses its published typed API, not our DeepSeek notebook prompt. Only state, question instructions and criteria are sent; gold/provenance are excluded. Returned native distributions are used unchanged. Choice/noul/score mappings are the pinned original upstream adapter's mappings.

Jev is 9 decisions ahead of DeepSeek-Dep (190/231), and 11 ahead of native Bonsai-Dep (188/231), on this single public pass. These runs were at different times and use different runtime/timing boundaries. They do not establish a general winner or official rank. The earlier upstream historical Jev 200/231 remains a separate run, not this measurement.

## Reproducible evidence

- Upstream pin: `bb05a335bc809e61b20c0f745d25499a82b326fc`.
- Dataset SHA-256: `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
- Records SHA-256: `30b3fd98a95857fbc9e7e45c0178d589fc62c6789c1af60f02219df9b13e116e`.
- Local run: `benchmark_runs/jev_original_public_20261006T200838+0700`; manifest, fsynced records, original request/response bodies and ledger preserved locally, not published.
- Source: [original TypeSafe adapter](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/jevbench/adapters/typesafe.py), same pinned runner/scorer. Downloaded files verified against pinned Git blob hashes.
- Orchestration: `benchmarks/jevbench_jev.py`; four offline tests, 100% line/branch coverage. Tests verify selective credentials, all three native mappings, no gold leakage, source integrity and a mocked end-to-end run.
- Only `TYPESAFE_API_KEY` selected from Boss's credentials file; no values, headers or credentials published. No services stopped/restarted and no DeepSeek/Bonsai rerun.

## Official scoring disclaimer

This is a legacy **231-case public diagnostic**, not the current official 1,500-decision evaluation. Official Capability, Composite and ranking are null. Organizer-run evaluation is a separate submission/payment gate.

## Bottom summary

- Original Jev: 199/231, 86.15%, all valid.
- Provider API round-trip: p50 418.09 ms, p95 516.34 ms.
- Tariff-derived total cost: $0.008874978, not invoice.
- Original adapter/native probabilities, no retries or zero-fill.
- Same public tasks and scorer as Dep's reports.
- Not an official leaderboard score or rank.
- No action needed from Boss for this completed run.
