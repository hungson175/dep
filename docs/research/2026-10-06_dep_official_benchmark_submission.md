# Dep (Dép): official benchmark submission research

Verified 6 October 2026, Vietnam time. Display name is **Dep**, not Z or Bonsai-Dep. Scope: identify the remembered $49/$99 service and the organizer-run native evaluation route. Read-only public research; no submission, payment, credentials, or model inference.

## Exact match

The [Benchmark Heaven priority evaluation page](https://benchmarkheaven.com/jev-models/request-evaluation) explicitly displays $49/$99 radio options. Independently checked in a fresh browser: selecting JevBench and larger-open-GPU pricing displays a $99 submission/payment button. No contact data was entered and the button was not clicked. The page links the newer unified [/submit](https://benchmarkheaven.com/submit) form. Static text extraction misses its conditional pricing controls; the client and server-delivered pricing also list 4900/9900 cents.

JevBench evaluates text-based typed decisions, not free-form conversational quality. Its [main board](https://benchmarkheaven.com/jev-models) ranks open-weight systems evaluated by the organizer. The observed current evaluation release is v1.6.1, board v1.7.0, with 1,500 decisions per system. Our historical 231-case accuracy is not that score.

## Prices and service

| Access | Fee per benchmark, before tax |
|---|---:|
| Hosted API or open model up to about 9B | $49 |
| Larger open model run on organizer GPUs | $99 |

Sources: [priority form](https://benchmarkheaven.com/jev-models/request-evaluation), [live terms](https://benchmarkheaven.com/terms).

**Inference for our configuration:** Dep using the existing Bonsai 27B weights, executed on organizer hardware, falls into $99 + applicable tax for JevBench alone. The 7.2 GB quantized file does not imply a 7.2B-parameter model. API-served evaluation is cheaper but is a different access/timing route and is not the user's requested native evaluation.

Payment buys priority, not altered scoring. Public runs publish aggregates with a priority-run marker; private runs produce a private report. Stripe handles checkout. Review occurs after payment. Results are promised within 48 hours, with pauses while awaiting information; requested corrections restart the full period after the update. Missed deadlines permit a refund request; unsafe or fairly unevaluable submissions are refused and refunded. Current live terms supersede search snippets that described automatic late refunds. [Terms](https://benchmarkheaven.com/terms)

## Recommended form selections

- Model name: **Dep**.
- Benchmark: **JevBench only**; no image/audio test for the current text-only system.
- Tier: larger open model on organizer GPUs, $99.
- Visibility: public leaderboard, consistent with Boss's ranking request.
- Access: open weights.
- Contact email: Boss must specify.
- Code: pinned commit in https://github.com/hungson175/dep.
- Weights: exact public Hugging Face repository/revision, GGUF filename and hash used by our adapter.
- Explain underlying Bonsai weights, decision adapter, quantization, prompts and public-set development exposure. Do not claim newly trained foundation weights.

The current GitHub library release is the **DeepSeek** adapter; it does not itself provide the submission-ready Bonsai native loader/container.

## Technical acceptance and remaining work

The organizer's [open-weight preparation instructions](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/ops/priority-evaluation/PREP-OPEN-WEIGHTS-PROMPT.txt) describe offline execution in a disposable GPU container, pinned image digest and code tree, pinned model revision/hashes, and a documented VRAM requirement no greater than 80 GB. Expected interface: `/v1/systemone`, or in-process Python `predict(state, questions)` returning their typed response shape. Reference-model pricing determines the cost axis, not zero local API spend.

The instructions mention safetensors; the [pod runner](https://raw.githubusercontent.com/fstandhartinger/model-market-comparison/main/ops/priority-evaluation/pod_runner.py) validates generic file hashes. This does not prove rejection or turnkey support for our GGUF/custom llama.cpp build. **Confirm native GGUF/custom-runtime acceptance and current evaluation release before paying.** Then package a reproducible adapter/image without silently swapping weights, precision or prompts.

Organizer contact shown on the priority page: **info@productivity-boost.com**. Proposed preflight: “Please confirm that Dep, an open decision system using Bonsai 27B GGUF with a pinned custom llama.cpp runtime, can be evaluated on your GPU under the current official JevBench release, with native inference timing and a public Dep row. We will provide pinned source, model hashes, container digest and typed interface. Is the fee $99 plus tax?” No message has been sent.

The generic submit form permits an encrypted key field while priority terms forbid credentials in requests and prescribe later encrypted handover. Our open-weight route needs no keys; do not submit secrets. [Priority page](https://benchmarkheaven.com/jev-models/request-evaluation), [terms](https://benchmarkheaven.com/terms).

## Bottom summary

- Exact $49/$99 site found: Benchmark Heaven.
- Correct public display name: Dep.
- Recommended route: $99 plus tax, JevBench, organizer GPU, public result.
- Official score requires their run; historical accuracy is only supporting evidence.
- Confirm GGUF/runtime support before payment and prepare a pinned runnable package.
- Need Boss's contact email and final checkout approval when proceeding.
- No submission, payment, credential disclosure or paid inference performed.
