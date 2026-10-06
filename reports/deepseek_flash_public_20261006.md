# DeepSeek Flash notebook — public JevBench run, 2026-10-06

## Result

- Completed **231/231** cases; **178 correct (77.06%)**.
- **218 valid distributions (94.37%)**; 13 missing-candidate failures count as wrong, alongside 40 valid-but-wrong decisions.
- API round-trip latency: **p50 683 ms; p95 912 ms** (serial calls from Boss's workstation).
- Peak-tariff cost upper bound: **$0.0411036 total**, **$0.1779/1,000 decisions**; not a provider invoice.
- Tokens reported by API: **136,088 input; 231 output**. No retries, transport failures, truncation, bias, or model fallback.

## Accuracy breakdown

| Case group | Correct / planned | Accuracy | Valid distributions |
|---|---:|---:|---:|
| easy | 45/48 | 93.75% | 45/48 |
| original | 64/72 | 88.89% | 69/72 |
| hard | 69/111 | 62.16% | 104/111 |

| Request type | Correct / planned | Accuracy |
|---|---:|---:|
| choice | 109/139 | 78.42% |
| noul | 55/74 | 74.32% |
| score | 14/18 | 77.78% |

## Probability diagnostics

- Accuracy among valid distributions only: 81.65%. This is not the headline accuracy: dropping 13 failed cases would conceal adapter failures.
- ECE (10 bins, top-label confidence): 0.1131; mean multiclass-sum Brier: 0.2800. Both exclude invalid distributions and use 218 valid cases.
- Score expected-value MAE: 0.2671 levels on 18 Score cases.
- Among the 178 valid cases with top-label probability at least 0.9, mean probability was 99.48%, but accuracy was 90.45%. The conditional token probabilities are overconfident on this sample; renormalization alone is not calibration.

## Historical comparison on exactly the same 231 public item IDs

| System | Correct / planned | Raw accuracy |
|---|---:|---:|
| DeepSeek one-token notebook (this run) | 178/231 | 77.06% |
| Jev 1.13.0 (TypeSafe AI) (historical) | 200/231 | 86.58% |
| DeepSeek V4.1 Flash (thinking default) (historical) | 226/231 | 97.84% |

These baselines come from the pinned upstream v1.3.0 per-task artifact. They are **historical, not fresh controlled reruns**: provider version, request strategy and runtime conditions can differ. The upstream DeepSeek route used thinking and verbalized JSON probabilities, unlike our first-token logprobs. This suggests a real quality cost for the one-token shortcut, but does not isolate its causal effect.
[Historical per-task source](https://raw.githubusercontent.com/fstandhartinger/jevbench/bb05a335bc809e61b20c0f745d25499a82b326fc/results/v1.2/jevbench-v1.2-per-task.json)

## Scope and reproducibility

- This is the **231 legacy authored public cases**, not the current v1.6 public/sealed pool. No official Intelligence, Capability, Composite, or leaderboard rank is claimed; 77.06% raw accuracy must not be confused with the post's benchmark score.
- Quyet was **not** run. No statement that this adapter beats Quyet is supported by this experiment. No Vietnamese translations or private cases were added.
- Live model-list identity check returned `id=deepseek-flash`, `name=DeepSeek-V4.1-Flash`.
- Recipe: `learn/deepseek-flash.ipynb`, implemented in `deepseek_flash.py`; thinking disabled, max_tokens=1, temperature=1, logprobs=True, top_logprobs=20. Missing candidates fail closed. Score prompts remain 1-based; only benchmark output keys shift to 0-based.
- [Upstream pin](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc): `bb05a335bc809e61b20c0f745d25499a82b326fc`. Files were verified against Git blob hashes; source and dataset hashes are in the run manifest.
- Run directory (git-ignored): `benchmark_runs/deepseek_flash_public_20261006T134655+0700`. It contains 231 raw responses, durable records, manifest, summary, and historical comparison artifacts. Authorization headers and keys are never written.
- Shared budget ledger remains below the approved $20 ceiling. Peak pricing was verified from [official DeepSeek docs](https://api-docs.deepseek.com/quick_start/pricing/) on 2026-10-06: $0.30/M cache-miss input, $0.006/M cache-hit input, $1.20/M output. Off-peak/holiday charges may be lower.
- Code validation: 31 offline tests passed; 99% total line coverage. The real 231-request run additionally exercises the actual API seam; it does not establish a production SLA or adversarial safety.
- No service restart, deployment, dependency installation, public submission, or Git push occurred.

## Bottom summary

- Status: full public authored run completed.
- Result: 178/231 correct (77.06%); hard tier 69/111 (62.16%).
- Reliability limitation: 13 missing-candidate failures (5.63%).
- Speed: p50 683 ms; p95 912 ms, API round trip.
- Cost upper bound: $0.0411036 total; $0.1779/1,000 decisions.
- Interpretation: below historical Jev and thinking-DeepSeek accuracy on these same item IDs; not an official current leaderboard comparison.
- Need from Boss: nothing; further experiments or official submission require a new assignment.
