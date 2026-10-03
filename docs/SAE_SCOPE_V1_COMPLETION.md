# Qwen-Scope SAE V1 collection status — 2 October 2026

This is a data-collection status report, not a feature interpretation. The primary laptop run is **partial for its requested 28-layer scope**. Its combined historical and new raw residual caches cover all 28 transformer blocks, while sparse SAE encoding is complete at nine layers. The original pilot, V3 runs, fixtures, reports and caches were read but not rewritten.

| Item | Verified coverage |
|---|---:|
| Exact V3 laptop cases | 6,322 cases; 12,644 case/readout associations |
| Comparison-question prefixes | 1,888 final-token associations, with candidate continuations retained |
| Unique main computational readouts | 14,500 after deduplication |
| Metadata-stratified token traces | 128 distinct prompts; 4,805 non-padding token positions |
| Existing V3 cache readout hits | 12,612 at blocks 4, 8, 12, 16, 20, 24, 28 |
| New raw-state shards | 8,194 main text shards; 128 trace shards |
| Fully SAE-encoded zero-based layers | **0, 1, 3, 7, 11, 15, 19, 23, 27** |
| Partially encoded layer | 2: 7,424 / 14,500 main rows; 0 / 4,805 trace rows |
| Other layers | 18 layers with no sparse chunks yet; their raw states are available |
| Aggregate sparse rows | 137,924 / 406,000 main layer-readouts; 43,245 / 134,540 trace layer-tokens |

The trace sample includes 18–19 prompts from each of the seven case families, selected deterministically from fixture metadata before looking at model or SAE results. `trace_selection.json` records exact selected and excluded IDs. A comparison prefix has a final-token readout; V3 defines no target-entity span for it, and that exclusion is recorded. Case metadata preserves targets, splits, wording, binding and nuisance contrasts, hue triplets and factorial identifiers. Physical units are joined through `target_registry.json`.

The SAE release is `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100` at revision `2756a49850d1d0288a18e561783deae0d810791e`, with 32,768 features and Top-K 100. Checkpoint SHA-256 values are pinned and verified per layer. The language model remains `Qwen/Qwen3-1.7B-Base` at `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. V3 block numbers are one-based; SAE layer indices are zero-based (`block - 1`). The smoke trace’s layer-3 entity state matched its historical cache vector exactly (maximum absolute difference 0). Encoding uses `TopK(ReLU(h @ W_enc.T + b_enc), K)` on original residuals in CPU float32 and saves selected IDs/values and reconstruction diagnostics, without storing dense latents or fitting probes.

The [smoke run](../runs/sae_scope_v1_smoke_top100-cb4a52cc096f/collection_report.md) is **complete** for its intended layer-3 scope: 568 unique main rows and 123 trace-token rows. The [laptop report](../runs/sae_scope_v1_laptop_top100-de55e3b9d113/collection_report.md) and [manifest](../runs/sae_scope_v1_laptop_top100-de55e3b9d113/manifest.json) give every layer’s expected, completed and missing counts, per-family coverage, technical reconstruction distributions, diagnostic flags, array shapes and file checksums. No non-finite or degenerate flags were recorded for encoded rows. These diagnostics are not semantic or predictive findings.

The laptop collection used 1,200.3 seconds of synchronized model-forward time and 4,800.1 seconds of cumulative SAE encoding time. The new run occupies about 2.2 GiB, excluding checkpoint downloads stored by Hugging Face. Approximately 2.3 GiB remained free at final validation. The 18 uncached SAE checkpoint files alone require another 9,666,226,804 bytes (about 9.0 GiB), so completing the 28-layer sparse dataset requires additional checkpoint storage or freeing space. The download preflight keeps a one-GiB reserve. The run has no recorded processing error; its partial state follows the explicit encoding-time and storage bounds.

Verification: 45 non-integration tests passed (one integration test deselected). The end-to-end smoke collection completed; its layer-3 trace/entity state matched the historical cache exactly. A final audit checked raw availability at every requested layer for all 14,500 unique main readouts and all 128 trace prompts. The final manifest checksum validation passed, and encoded rows had only zero-valued diagnostic flags.

From the project root, after `source .venv/bin/activate`:

```bash
pytest -m 'not integration' -q
qd sae validate --profile smoke --resume
qd sae validate --profile laptop --resume
qd sae plan --profile laptop --resume
```

After providing enough checkpoint storage, continue the existing raw-complete run without loading the language model:

```bash
qd sae encode --profile laptop --run-dir runs/sae_scope_v1_laptop_top100-de55e3b9d113 --resume --max-encoding-minutes 110
qd sae validate --profile laptop --resume
```

`--max-encoding-minutes` is a cumulative cap; 110 extends the completed 80-minute pass by 30 minutes. If validation remains partial, increase the cap again. Use [SAE_SCOPE_V1.md](SAE_SCOPE_V1.md) for the full schema, setup, optional extended profile, Top-50 variant and loader example. No causal-inner-product or scientific-analysis branch was modified.
