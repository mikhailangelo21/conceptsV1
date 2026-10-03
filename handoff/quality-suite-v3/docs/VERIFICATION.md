# Execution and verification log

Host: actual Apple Silicon Mac, macOS 15.2 / Darwin 24.2.0, arm64, Python 3.13.3. Native MPS was detected. No remote Linux/MPS claim was made.

- Installed and inspected PyTorch 2.14.0 and Transformers 5.17.0; all installed versions are pinned in requirements-lock.txt.
- Inspected Qwen3DecoderLayer.forward, Qwen3Model.forward and Qwen3ForCausalLM.forward. Complete decoder blocks return tensors; final RMSNorm is outside the block stack; logits_to_keep is supported.
- Initial offline suite: 19 passing tests, one real-model test deselected. Covers hook semantics/storage, causal prefixes, cleanup, observation/zero invariance, nonzero injection and downstream changes, immutable weights, exact scoring, counterbalancing, equal control norms, grouped uncertainty, label permutations, leakage, provenance imports, cache interruption/mismatch, complete synthetic workflow/resume, and native tiny-Qwen3 MPS float16.
- Checked all 780 geometry and 192 behavioral prompt boundaries using the actual pinned public tokenizer. Fixed a byte-level-BPE newline/punctuation merge assumption; the stable Answer: token suffix is asserted.
- Tiny random-model full-workflow test produced all five plot families and PNG/SVG outputs. These are temporary software-QA fixtures, not research results. Visually inspected its PCA rendering while the public model downloaded.
- Public model/tokenizer revision resolved to ea980cb0a6c2ae4b936e82123acc929f1cec04c1, 28 blocks × 2048 residual width, checkpoint bfloat16. Public weights downloaded successfully (3,441,185,608 bytes).
- Environment repair: macOS marked Desktop virtual-environment dependencies dataless; clean CLI imports stalled at file reads, confirmed by faulthandler and process sampling. Recreated the exact snapshot outside Desktop at ~/.local/share/quality-dimensions-llm/venv and linked the project's .venv to it. No user files were removed.

Real smoke/pilot status and final test totals will be appended after execution. No pretrained result is claimed by the offline checks above.

- Recreated-environment check: 20 offline tests passed (one integration test deselected), including a clean-process CLI launch regression check. `qd doctor` and `qd prepare-data` executed successfully.
