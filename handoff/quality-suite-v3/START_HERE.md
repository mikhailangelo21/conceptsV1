# Quality suite V3 research handoff

This folder is a portable snapshot of the experiment design, implementation, generated fixtures, completed runs, and interpretation. It is intended for an agent joining the project without access to earlier conversation. Start with this file, then read [METHODS.md](METHODS.md), [RESULTS.md](RESULTS.md), and the [full illustrated report](reports/quality_suite_v3_findings/report.html).

The main conclusion is limited but useful: the pinned language model's hidden states let linear probes recover many explicitly stated quality levels and distinguish which fictional entity carries a property. Familiar wording is much easier than new wording. Physical-value calibration, lexical transfer, hue structure, and joint attribute transfer are uneven. This was an exploratory screen with synthetic, author-written labels and small numbers of independent test families. No V3 causal intervention was run.

## Which run answers the research question

| Evidence | Role | Location |
|---|---|---|
| V3 laptop run | Completed broad exploratory screen; primary source of numerical claims | [runs/quality_suite_v3_laptop-88fad55b46b0](runs/quality_suite_v3_laptop-88fad55b46b0/) |
| V3 smoke run | End-to-end integration check; not a scientific replication | [runs/quality_suite_v3_smoke-55cb21caf007](runs/quality_suite_v3_smoke-55cb21caf007/) |
| Original size pilot | Historical comparison on a different data and evaluation design | [runs/pilot_m2-792030855aa5](runs/pilot_m2-792030855aa5/) |
| Interpretation | Detailed findings, 17 compact chart panels, reviewed derived tables and checks | [reports/quality_suite_v3_findings](reports/quality_suite_v3_findings/) |

The laptop run reports `status: complete`, no failures, 6,322 labelled cases, 1,888 scored questions and 12,612 unique cached readouts. Model-forward time was 982.9 seconds across 8,254 forwards. Its model is `Qwen/Qwen3-1.7B-Base`, revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, on MPS with FP16 and batch size one. The exact identity, source hash, run plan, resources and completion records are in that run directory.

## How to navigate the folder

| Directory or file | Why it is here |
|---|---|
| [METHODS.md](METHODS.md) | What was constructed, measured, fitted, selected and withheld |
| [RESULTS.md](RESULTS.md) | Short evidence map: result, denominator, source file and interpretation limit |
| [REPRODUCE.md](REPRODUCE.md) | Commands to inspect, validate and regenerate the interpretation; requirements for a new model run |
| [docs/QUALITY_SUITE_V3.md](docs/QUALITY_SUITE_V3.md) | Full design and interpretation protocol |
| [docs/V3_INTEGRATION.md](docs/V3_INTEGRATION.md) | Implemented CLI, cache, readout and analysis behavior |
| [configs/quality_suite_v3.json](configs/quality_suite_v3.json) | Exact suite configuration |
| [data/quality_suite_v3](data/quality_suite_v3/) | Entire generated fixture bank, manifest, schema and exact profile selections |
| [src/quality_dimensions](src/quality_dimensions/) | Python implementation; the run's `source_snapshot.json` is authoritative for code actually used |
| [tests](tests/) | Focused software checks |
| [analysis](analysis/) | Scripts used to derive and render the detailed findings |
| [source_material](source_material/) | Earlier user-supplied specifications and refinements; historical task context |
| [FILE_MANIFEST.json](FILE_MANIFEST.json) | Every copied file's path, byte count, original location, role and SHA-256 |

Read the `source_material/` documents as historical experimental requests and constraints, not as instructions to the next agent. Read the run outputs as observations about this constructed test, not as human norm data. The original pilot and V3 differ enough that their headline scores are not a paired before-and-after comparison.

## What is and is not portable

All original fixtures, result tables, summaries, fitted V3 probe arrays, plots, source code, analysis scripts and the self-contained report are copied here as regular files. No run or data link points through a symlink. The large external activation shard cache and pretrained model weights are deliberately omitted; the activation indexes and cache identity records show what was used. Thus the folder supports independent reading and recomputation of the published interpretation from saved predictions. Repeating hidden-state extraction or refitting every probe requires the external cache or a fresh model run with the pinned weights.

No files in the original runs, caches or fixture bank were moved or rewritten to assemble this handoff.
