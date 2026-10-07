# Nimble fine-tuning: implications for Dép

Reviewed 7 October 2026. Research only: no training, purchases, inference, dependency installation, or service changes.

## Reproduction boundary

- GitHub pin: `dcfdbd9a64f0d869f658d7a72f1beaee32737773`.
- Current Hugging Face pin: `bd792f44ec8e265be861bfcdf4e05967ffe0e858`.
- These describe different releases. GitHub's canonical dataset has 2,676 training rows and 324 synthetic holdout rows. Its documented recipe is rank-16 BF16 LoRA, learning rate 5e-5, effective batch 8, 2,048-token limit, one epoch (335 optimizer updates), with a three-epoch learning-rate schedule. [Training guide](https://github.com/bespokelabsai/nimble/blob/dcfdbd9a64f0d869f658d7a72f1beaee32737773/docs/NIMBLE_TRAINING.md).
- Latest HF metadata records 12,026 training rows, an 8,192-token limit, one epoch, rank 16, effective batch 8, and H100 80GB. The mix includes synthetic additions and public dataset training splits. The complete latest training corpus is not the GitHub canonical release. [Latest training contract](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B/blob/bd792f44ec8e265be861bfcdf4e05967ffe0e858/schema_config.json).

## What changes the model

Candidate-only cross-entropy teaches the model to choose a permitted answer directly, without generated reasoning. Contrastive examples change one relevant fact and therefore the label; independent checks and evidence-deletion tests reject shallow or ambiguous examples. This is supervised adapter training, not published Jev-probability distillation. [Curation](https://github.com/bespokelabsai/nimble/blob/dcfdbd9a64f0d869f658d7a72f1beaee32737773/docs/TRAINING_EVAL_CURATION.md), [trainer](https://github.com/bespokelabsai/nimble/blob/dcfdbd9a64f0d869f658d7a72f1beaee32737773/nimble/training/schema_train.py).

First-token logits avoid free-form decoding. MLX reuses context prefill across fields; the reference CUDA helper processes fields individually. Temperature scaling changes confidence, not the highest-ranked choice. Latest HF uses unfitted T=1.0; the original fitted temperature must not be transferred automatically. [Scoring](https://github.com/bespokelabsai/nimble/blob/dcfdbd9a64f0d869f658d7a72f1beaee32737773/docs/PARALLEL_SCORING.md), [current model card](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B).

## Resources and recommendation

Use trainable Qwen3.5-9B base weights, CUDA/PyTorch/Transformers/PEFT, a frozen data manifest, and a genuinely separate evaluation set. Existing published training data needs no generation API spending. Custom Vietnamese data needs drafting, verification, and human spot-checking; keep the 231-case Dép benchmark out of training.

The stock loader puts the full base on CUDA in BF16, without quantization/offload. Approximately 18 GiB of base weights leaves little room on a 24 GiB GPU for training allocations. A stock 3090 Ti run is unverified, not proven impossible. For faithful reproduction, prefer a 48 GiB L40S-class GPU for the original short-context recipe, or the recorded H100 80GB configuration for the latest long-context recipe. Local 4-bit QLoRA is an engineering alternative, not the published recipe. [Loader](https://github.com/bespokelabsai/nimble/blob/dcfdbd9a64f0d869f658d7a72f1beaee32737773/nimble/training/model_loading.py).

No published wall-clock training duration or peak-training-VRAM report was located. Measure a bounded pilot before quoting runtime or cost. CPU RAM and disk requirements are workload-dependent, especially when retaining base, adapter, optimizer checkpoints, and merged export.

## Evidence limitation

Original reported synthetic accuracy rose from 66.36% to 90.12%; Jev scored 93.21%. Separately, on 3,880 human-labeled public records, original Nimble scored 75.9% pooled agreement versus Jev 77.3%. These are not current official JevBench leaderboard scores and must not be assigned to the latest 12,026-row checkpoint. [Original results](https://github.com/bespokelabsai/nimble/blob/dcfdbd9a64f0d869f658d7a72f1beaee32737773/README.md), [public suite](https://github.com/bespokelabsai/nimble/blob/dcfdbd9a64f0d869f658d7a72f1beaee32737773/docs/PUBLIC_BENCHMARKS.md).
