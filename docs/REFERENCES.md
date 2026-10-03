# Implementation sources consulted

Consulted 2026-09-27; installed implementation inspected directly rather than trusting moving main.

- https://huggingface.co/Qwen/Qwen3-1.7B-Base — dense pretrained base model; no chat-template substitution.
- https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json — architecture checked against the loaded revision.
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py — installed Transformers 5.17.0 inspected and offline tested; complete block returns a tensor; final RMSNorm is separate; causal LM supports logits_to_keep.
- https://docs.pytorch.org/docs/stable/notes/mps.html — MPS runtime; actual backend detection and synchronization used.
- https://proceedings.mlr.press/v235/park24c.html — motivation for distinguishing linear directions and geometric assumptions.
- https://arxiv.org/abs/2309.16042 — intervention metrics and methodology cautions.
- https://arxiv.org/abs/2404.15255 — interpretation of activation interventions.

The software does not claim this pilot exactly replicates any paper or establishes their conclusions.
