# DeepSeek Flash — zero-fill offline replay, 2026-10-06

## Result

Reprocessed the **same 231 saved public responses** with missing options set to
0.0 and the available candidates renormalized. No new requests, new prompts,
model tuning, or API spend occurred. This is a post-hoc decoding-policy change,
not a fresh benchmark run or an official leaderboard result.

| Policy | Valid / planned | Correct / planned | Accuracy |
|---|---:|---:|---:|
| Original strict | 218/231 | 178/231 | 77.06% |
| New default: zero-fill | 231/231 | 190/231 | 82.25% |

All **13** previously failed decisions now produce complete normalized maps;
**12** become correct and **1** remains wrong. The distributions and predictions
for the original 218 valid cases are unchanged. The original strict evidence
and [report](deepseek_flash_public_20261006.md) remain untouched.

| Request type | Correct / planned | Accuracy |
|---|---:|---:|
| choice | 119/139 | 85.61% |
| noul | 57/74 | 77.03% |
| score | 14/18 | 77.78% |

Zero is an approximation for an option omitted from the top-20 report, **not a
measured zero probability**. Confidence may increase; calibration is not proven.
When all candidates are absent or the report is malformed, decoding still raises
an error. `missing_policy="error"` retains the original strict behavior.

## Verification and evidence

- Both the benchmark adapter and the public `dep_deepseek.DeepSeek.predict` API
  replayed all 231 saved responses and returned identical distributions after
  canonical label mapping. A guard rejected any attempted network request.
- Every prepared request matched the original stored request exactly. Every
  missing option returned 0.0; all 13 recovered maps passed upstream validation.
- Scope remains the pinned **231 legacy public authored cases**, not the current
  public/sealed leaderboard pool; official score and rank remain unset.
- Upstream pin: `bb05a335bc809e61b20c0f745d25499a82b326fc`.
- Original records SHA-256:
  `c3fdc37233827ec55bdaf69acd0c4e33344092d5bd132363acd22c9ef8c6f3be`.
- Original `records.jsonl`, `summary.json`, and `manifest.json` hashes were checked
  before and after replay; none changed.
- Derived records and summary are separate, git-ignored artifacts in
  `benchmark_runs/deepseek_flash_zero_fill_replay_20261006/`.
- A sanitized recorded missing-option response is retained as an offline
  regression fixture in `tests/fixtures/deepseek_missing_options.json`.
- Validation: 47 offline tests passed, including isolated wheel installation
  and HTTP-to-benchmark-record integration; 98% combined statement/branch
  coverage, with 100% coverage of the public wrapper.
- No credential files were read and no services, deployments, submissions,
  or Git remotes were changed during replay.

## Bottom summary

- Status: offline zero-fill replay complete; no new API calls or cost.
- Result: 190/231 correct (82.25%), versus the original strict 178/231.
- Recovery: all 13 failures now valid; 12 correct, 1 wrong.
- Preservation: original evidence and all originally valid distributions unchanged.
- Caveat: censored-probability approximation; not calibrated or an official rank.
- Need from Boss: nothing for local use; publishing remains a separate approval.
