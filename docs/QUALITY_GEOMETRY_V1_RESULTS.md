# Completed covariance geometry comparison

The implementation, smoke run and full-vocabulary laptop run are complete. Start with the [findings package](../reports/quality_geometry_v1_findings/START_HERE.md) and [interpretation](../reports/quality_geometry_v1_findings/findings.md).

Covariance geometry did not consistently improve the measured quality representations. Familiar-wording ridge averages improved slightly; held-out wording correlation fell from 0.543 to 0.498. Mean held-out MAE improved from 0.425 to 0.356, but only 5 of 14 qualities improved and 9 worsened. The V3-style standardized ridge at the same final site remained stronger in aggregate. Output-word coverage is too small for a broad conclusion, and contextual consistency/selectivity changes are mixed.

- Full run: `runs/quality_geometry_v1_laptop-e95e9cf10513` — 6,322 cases, uniform covariance of all 151,936 output rows, unregularized condition number 1,776.5.
- Successful smoke: `runs/quality_geometry_v1_smoke-bbedd7380550` — 260 cases, separately labelled 8,192-row sampled covariance.
- Verification: 41 offline tests passed; real smoke and laptop checks each reproduced all output logits on eight prompts with zero observed difference; 3,258 saved metric cells independently recomputed; 6,306 reused activation shards rehashed.
- No steering was run. Historical fixtures, runs, caches and V3 handoff were preserved.

See [implementation and reproduction instructions](QUALITY_GEOMETRY_V1.md). Exact tables, denominators and caveats are in the findings package. These exploratory authored fixtures do not establish causal or universal quality dimensions.
