# Dép — Luna 6 Decisions API public benchmark

15 October 2026. Native API distributions, **not the earlier subscription label-only run**. No model fine-tuning or manual logit-bias trick is used here.

## Result

**204/231 correct (88.31%)** on the same pinned legacy public cases. All 231 were attempted once, serially, without retry or model fallback. There were **228 valid native distributions** and **3 HTTP 503 failures**, counted wrong rather than removed from the denominator. No distributions were renormalized or zero-filled.

| Type | Correct | Accuracy | Valid distributions |
|---|---:|---:|---:|
| Choice | 121/139 | 87.05% | 137/139 |
| Noul | 67/74 | 90.54% | 73/74 |
| Score | 16/18 | 88.89% | 18/18 |

- Brier mean: **0.1410736842**, over 228 valid distributions.
- Top-label ECE: **0.0223245614**, over 228 valid distributions, using upstream bins.
- Ordinal expected-value MAE: **0.1783333333**, over the 18 Score cases.
- Provider API round-trip: **p50 464.31 ms**, **p95 958.55 ms**, including all 231 attempts. This is not native GPU compute latency.
- API reported **zero output tokens and zero reasoning tokens** on all 228 successful requests.

The failed tasks were `hard-sol-a-trap-10`, `hard-sol-c-judge_hard-10`, and `hard-sol-c-multi_hop-09`. All failed with HTTP 503 and no usable usage/distribution. They were not retried.

## Cost

Successful replies reported **150,608 input tokens**. Applying the published standard input price, $0.10/M, gives **$0.0150608** for known usage; this is a tariff-derived upper bound, not a provider invoice. The total is **unknown** because the three failed requests supplied no usage.

The durable $20 budget ledger retained $0.21 reservations for each unknown failed request. Its $0.6450608 accounting total is **not a billed charge**. Cached input is not discounted by the conservative estimator. [Decisions pricing](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability).

## What was measured

The official paid endpoint is `POST https://api.openai.com/v1/decisions`, requesting `gpt-6-luna`. The adapter forwards only canonical task state, question instructions, and criteria:

- **Noul → predicate:** native probability of true becomes `yes=p`, `no=1-p`. Supplied true/false definitions are appended to instructions.
- **Choice → choice:** original option values and descriptions are passed as typed choices. The returned distribution is keyed by those original values.
- **Score → score:** rubric levels retain their original order and use zero-based labels. Returned level probabilities feed the upstream scorer.

The scorer uses the native distribution, not generated answer text or verbalized percentages. No `max_tokens`, `logit_bias`, sample-count estimate, or fabricated one-hot distribution is involved. This measures OpenAI's native Decisions interface behind a Dép task adapter; it does not demonstrate the manual one-token-logit recipe on Chat Completions. Native probabilities are model estimates, not a guarantee of calibration outside this small diagnostic set.

A separate capability probe successfully returned **all 35 option probabilities**, summing to 1.0. That probe is not one of the 231 scored cases. Thirty-five is a verified supported count, not an established maximum. Separate paid Chat Completions probes rejected `top_logprobs=20`/`35` with a reported maximum of 5 for this model, and rejected `logit_bias` as unsupported. Those model/endpoint-specific observations must not be generalized to every GPT API.

## Comparison and scope

Earlier single passes on the same 231 cases: original Jev **199/231 (86.15%)**; zero-fill DeepSeek **190/231 (82.25%)**; native Bonsai **188/231 (81.39%)**. The separate Luna subscription label-only pass scored **212/231 (91.77%)**, but supplied no candidate probability distribution. Dates, interfaces, prompt packaging and failure behavior differ; these figures do not establish an overall model ranking.

This is a **legacy public diagnostic**, not the current private/official JevBench evaluation. Official Capability, Composite and leaderboard rank remain null.

## Reproduction and audit

```sh
python -m benchmarks.jevbench_luna_decisions
```

Requires a valid personal `OPENAI_API_KEY` in `~/dev/.env`. Importing the adapter reads no credentials and makes no requests. Only that selected key is loaded when the CLI is invoked. No credential headers are saved, published, or forwarded on redirects. No deployed service was changed.

- Upstream pin: `bb05a335bc809e61b20c0f745d25499a82b326fc`.
- Dataset SHA-256: `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
- Records SHA-256: `f72d42b9490845ed91d5c0e92e3e66dd9870fb1e9811067bbd861b9e95e42235`.
- Local evidence: `benchmark_runs/luna6_decisions_20261015T032649+0700`, with manifest, durable per-task records, raw responses, summary and independent audit.
- Audit reconstructed every request from state/question-only inputs, checked all native model/usage/distribution fields, re-scored all 231 results, and verified both implementation hashes.
- Full suite: **120 tests PASS**, four pre-existing opt-in tests skipped. New Decisions code: **97.97% combined coverage**, 99.36% statements, 92.50% branches. VCR replay covers a real 35-option API response. See [`../../coverage_report.md`](../../coverage_report.md).
- Public aggregate data: [`luna6_decisions_public_20261015.json`](luna6_decisions_public_20261015.json). No raw private run data or credentials are published.

The benchmark HTML page was not changed; this report is published separately to prevent conflating native-distribution and label-only results.

## User-authorized retry of the three HTTP 503 cases

15 October 2026: Boss explicitly requested rerunning only the three failed requests. Each was sent once more; **all three returned HTTP 200, valid native distributions and correct answers**, with zero output/reasoning tokens. The original run, raw evidence and 204/231 headline above remain unchanged.

Replacing those three unresolved cases with their new replies gives **207/231 correct (89.61%)**, with 231 valid distributions. This is a **retry-resolved result**, not a fresh no-retry 231-request pass. There were 234 provider requests overall: the original 231 plus three authorized retries.

Retry usage-derived charge: **$0.0003097**, not an invoice. Initial failed-request charges remain unknown. Any latency in the retry-resolved aggregate describes the selected replies only and excludes the initial failed attempts; do not label it end-to-end retry latency or silently replace the original latency measurement.

The original 228 successful cases were not rerun. Local supplemental evidence: `benchmark_runs/luna6_decisions_retry3_20261015T033223`, including retry records, raw replies, resolved summary and audit. Public JSON includes a separate `user_authorized_retry` section, preserving the original summary.
