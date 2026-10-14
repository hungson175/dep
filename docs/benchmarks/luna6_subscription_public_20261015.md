# Dép–Luna 6: subscription public benchmark

15 October 2026. **Label-only prototype, reasoning disabled; not a complete probability-scoring adapter.**

## Result

**212/231 correct (91.77%)**. All 231 responses were valid, HTTP 200, with returned model `gpt-6-luna` and **zero reasoning tokens**. One request per case, serial, no retries or fallback models.

| Type | Correct | Accuracy |
|---|---:|---:|
| Choice | 124/139 | 89.21% |
| Noul | 72/74 | 97.30% |
| Score | 16/18 | 88.89% |

Subscription Responses round-trip latency: **p50 1.298 s**, **p95 1.664 s**. This includes request preparation, network, provider execution and local parsing. It is not native GPU compute time. Usage: 138,286 input tokens, 1,155 output tokens, zero reasoning tokens. Subscription quota was used; no paid API key was used. Dollar cost is unknown, not zero.

## Access and probability limitations

The notebook's OmniRoute login rejected Luna 6 as unsupported. Direct Codex CLI using the owner's existing ChatGPT login succeeded with reasoning disabled. A separate read-only client using that same owner login then succeeded against `https://chatgpt.com/backend-api/codex/responses`. No proxy installation, credential copying, service restart or manual authentication refresh was performed. This backend client is experimental and owner-local, not a public demo API or an entitlement bypass.

Live parameter probes on the working subscription endpoint returned HTTP 400 for both `top_logprobs` and `logit_bias`. Requesting `message.output_text.logprobs` worked, but returned only the selected token's logprob with an empty alternative-token list. Therefore no full candidate distribution was available.

The benchmark scores the exact returned answer mark with upstream `score_label`. It does not manufacture a one-hot probability map, confidence, Brier, ECE, or ordinal expected-value MAE. All those probability metrics remain null. Unlike the old Luna notebook, this is not a 20-sample frequency estimate.

The user prompt and label mappings are reused from `deepseek_flash.build_request`; the subscription request adds the short instruction `Classify the supplied context. Return only the requested answer code.` It requests `reasoning.effort=none`, `store=false`, and no tools. Server-reported defaults on the successful probe were temperature 1.0 and top-p 0.98. Token-limit parameters are not supplied because this subscription interface does not expose the one-token limit used by DeepSeek. Output is accepted only when the complete answer exactly matches a requested mark.

## Comparison scope

Same legacy 231-case public dataset and label-accuracy denominator as the earlier Dép reports. Earlier runs: original Jev 199/231 (86.15%), zero-fill DeepSeek 190/231 (82.25%), native Bonsai 188/231 (81.39%). Luna's result is ahead on this single public pass, but dates, interfaces and timing boundaries differ. It does not establish an overall winner. Probability-quality comparisons are unavailable for this label-only run.

This is **not the official JevBench private/current evaluation**, Capability, Composite, or a leaderboard ranking.

## Reproduction and evidence

```sh
python -m benchmarks.jevbench_luna
```

Requires the owner's existing Codex ChatGPT login, not an OpenAI API key. Imports do not read credentials or make calls. Failed access/rate-limit responses stop immediately; three consecutive failures stop the run. No automatic refresh, retry or fallback.

- Upstream pin: `bb05a335bc809e61b20c0f745d25499a82b326fc`.
- Dataset SHA-256: `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
- Records SHA-256: `f8f4e9e1b35c1f5924dd94522990a68fe784c49ce53e9ce7a2fdaf304aa13107`.
- Private run directory: `benchmark_runs/luna6_subscription_20261015T030508+0700`.
- Audit verified all 231 raw requests against canonical prompt construction, each returned model/status/reasoning count, each scored label, and both implementation hashes.
- Full suite: 114 tests passed, four pre-existing opt-in tests skipped. New-code combined coverage 96.13%; statement coverage 97.93%, branch coverage 88.89%. See `coverage_report.md` and generated `htmlcov/index.html`. VCR replay matches method, URL and request body; credential headers are excluded.

No live site deployment or HTML report layout was changed. Public aggregate summary: [`luna6_subscription_public_20261015.json`](luna6_subscription_public_20261015.json).
