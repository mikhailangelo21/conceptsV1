# V3 findings deliverable

Open **report.html** for the complete self-contained report, with seventeen compact chart panels and three tables. It needs no server or model inference. **findings.md** is the editable narrative; **artifact.json** is the complete report manifest and bounded data snapshot.

The report distinguishes decodability, binding, numerical calibration, generalization and causal evidence. It includes limitations, comparison with the original pilot, and a prioritized next experiment.

## Validation

- Recomputed 564 scalar prediction partitions; maximum MAE round-trip difference: 2.22e-16.
- Reconciled all 1,888 behavioral records with saved summaries.
- Verified 48 matched underlying natural/A-B comparisons, each with four A/B counterbalances.
- Checked IDs, denominators, entity coverage and physical training ranges.
- Canonical HTML delivery: validation, packaging and browser verification passed.
- Browser checks passed at 1440px and 390px, including chart/table visibility, overflow, and the source dialog interaction.
- Seventeen compact chart panels, three tables and 36 report blocks. Static chart exports are embedded by the canonical builder where its size budget permits; every chart also has an accessible data table.

## Visual design

- Long property comparisons use horizontal bars with exact value labels; every property name is visible.
- Familiar and reworded conditions keep the same series order across figures.
- Ordering charts explain the −1 to 1 scale; accuracy charts mark the 50% guessing reference.
- Error charts state that lower is better and mark either one ordinal level, the word-only binding baseline, or the 90° uniform-angle reference as appropriate.
- Physical errors are normalized to each property's training range and displayed as percentages.
- Dense 14–18-property figures are split into small panels without filtering any result; factorial panels follow meaningful domain groups.
- `chart_map.json` records the question, metric, row count, category count and visual encoding for every panel.

The native renderer returned a payload without an accessible report link, so HTML is the final delivery surface. Long identifiers are abbreviated in the HTML prose; the complete model revision and run identifier remain in findings.md and the original run artifacts.

## Reproduce

From the project root:

```bash
.venv/bin/python analysis/v3_findings.py
.venv/bin/python analysis/v3_report.py
node /Users/popthornthanatit/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs --input reports/quality_suite_v3_findings/artifact.json --output reports/quality_suite_v3_findings/report.html
```

These commands regenerate report outputs only. Original datasets, run results and caches are read-only inputs. `provenance.json` fingerprints consumed CSV inputs; `validation.json` records analytical checks. Supporting CSVs and `datasets.json` retain full precision beyond the rounded narrative.
