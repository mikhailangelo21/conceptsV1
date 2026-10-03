"""Create a concise, source-backed HTML-report manifest; no model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from geometry_v1_diagrams import add_diagrams


def build(out: Path) -> None:
    def read_csv(name):
        with (out / name).open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key, value in row.items():
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
        return rows

    controls = read_csv('ridge_controls.csv')
    detail = read_csv('heldout_per_quality.csv')
    assert len(controls) == 6 and len(detail) == 14
    assert all(r['n_rows'] == 10 and r['n_groups'] == 2 for r in detail)
    assert sum(r['mae_delta'] < 0 for r in detail) == 5
    assert sum(r['mae_delta'] > 0 for r in detail) == 9
    assert sum(r['rho_delta'] < 0 for r in detail) == 8
    for row in detail:
        assert abs(row['mae_delta'] - (row['mae_covariance'] - row['mae_euclidean'])) < 1e-12
        assert abs(row['rho_delta'] - (row['rho_covariance'] - row['rho_euclidean'])) < 1e-12
    for method in ['covariance', 'euclidean']:
        summary = next(r for r in controls if r['method'] == 'ridge_' + method and 'paraphrase' in r['partition'])
        for metric in ['rho', 'mae']:
            assert abs(summary[metric] - statistics.mean(r[metric + '_' + method] for r in detail)) < 1e-12

    stamp = datetime.now(timezone.utc).isoformat()
    title = 'A new ruler, mixed results'
    manifest = dict(version=1, surface='report', title=title,
                    description='A concise guide to the completed covariance extension of the quality-dimensions experiment.',
                    generatedAt=stamp, blocks=[], charts=[], tables=[], sources=[])
    snapshot = dict(version=1, status='ready', generatedAt=stamp, datasets={})
    prefix = 'reports/quality_geometry_v1_findings/'

    def source(sid, filename, label, description, definitions=()):
        manifest['sources'].append(dict(
            id=sid, label=label, path=prefix + filename,
            query=dict(description=description, engine='local files',
                       tables_used=[prefix + filename],
                       metric_definitions=list(definitions),
                       filters=['Completed full-vocabulary laptop run ending e95e9cf10513; no smoke results pooled.'])))

    source('findings', 'findings.md', 'Reviewed experiment design and findings',
           'Detailed interpretation of the preserved V3 extension, including run identity, model, readout, controls, limits and original evidence paths.')
    source('ridge', 'ridge_controls.csv', 'Three readouts at the same model location',
           'All six method-by-wording rows. Equal-weight means over 14 scalar properties; each cell uses 10 examples from two held-out entity families per property. Derived by analysis/geometry_v1_findings.py; HTML display labels are assigned by analysis/geometry_v1_html.py.',
           ['rho: arithmetic mean of per-property Spearman rank correlations between predicted and authored levels; higher is better.',
            'mae: arithmetic mean of per-property mean absolute prediction error in authored 0–1 quality-code units; lower is better.',
            'Euclidean and covariance use centered ridge without per-coordinate standardization. Standardized uses V3-style StandardScaler plus ridge at the same final post-normalization token site.'])
    source('detail', 'heldout_per_quality.csv', 'New-wording results for every property',
           'All 14 properties, sorted by covariance-minus-Euclidean error. No property filtering. Two independent test families and 10 examples per property.',
           ['mae_delta = mae_covariance − mae_euclidean. Negative means lower error with covariance.',
            'rho_delta = rho_covariance − rho_euclidean. Positive means better rank ordering.',
            'Counts compare signs of the unrounded deltas; median is taken across the 14 property-level error deltas.'])
    source('consistency', 'context_consistency.csv', 'Alignment of changes inside the model',
           'Family-balanced alignment of adjacent authored level changes. Familiar and paraphrased conditions each include 14 properties and 112 contrasts from two families per property; numeric condition has eight properties and 64 contrasts.',
           ['mean_cosine: direction alignment; +1 same direction, 0 perpendicular, −1 opposite.'])
    source('binding', 'binding_nuisance_summary.csv', 'Object assignment and irrelevant edits',
           'Simple context-contrast projectors, not ridge predictions or generated-answer accuracy. Binding/reordering: one test family; irrelevant edits: two.',
           ['fraction_positive: fraction of binding changes with the intended positive sign, summarized equally over properties.',
            'absolute_reorder_change and absolute_irrelevant_change: average absolute calibrated score movement under edits intended to preserve the measured level.'])
    source('factorial', 'matched_factorial_summary.csv', 'Changing one property while measuring the others',
           'Exactly 270 matched contrasts per wording and method, from one test family. The report retains both target and off-target quantities.',
           ['target: mean signed calibrated change in the property deliberately increased.',
            'off_target: mean absolute calibrated change across other measured scalar properties, first averaged within contrast then within family. Signed and absolute statistics are not interchangeable.'])

    # Execute the material source queries against saved raw run tables, then
    # reconcile to the reviewed CSVs. Display reshaping is separate from provenance.
    run_name = 'quality_geometry_v1_laptop-e95e9cf10513'
    run = out.resolve().parents[1] / 'runs' / run_name
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    for name, filename in [('run_metrics', 'metrics.csv'), ('run_ridge_comparison', 'ridge_comparison.csv')]:
        with (run / filename).open(newline='') as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            db.execute('CREATE TABLE ' + name + ' (' + ','.join('"' + c + '" TEXT' for c in columns) + ')')
            db.executemany('INSERT INTO ' + name + ' VALUES (' + ','.join('?' for _ in columns) + ')',
                           [[row[c] for c in columns] for row in reader])
    ridge_sql = '''SELECT partition, method, COUNT(*) AS n_qualities,
       AVG(CAST(rho AS REAL)) AS rho, AVG(CAST(mae AS REAL)) AS mae,
       MIN(CAST(n_rows AS INTEGER)) AS n_rows_per_quality,
       MIN(CAST(n_groups AS INTEGER)) AS n_families_per_quality
FROM run_metrics
WHERE method IN ('ridge_euclidean', 'ridge_covariance', 'ridge_v3_scaled_same_site')
  AND partition IN ('entity_test|graded|plain', 'entity_test+template|graded|paraphrase')
GROUP BY partition, method'''
    detail_sql = "SELECT * FROM run_ridge_comparison WHERE partition = 'entity_test+template|graded|paraphrase'"
    sql_controls = [dict(r) for r in db.execute(ridge_sql)]
    assert len(sql_controls) == len(controls)
    for expected in controls:
        actual = next(r for r in sql_controls if r['partition'] == expected['partition'] and r['method'] == expected['method'])
        for field in ['rho', 'mae', 'n_qualities', 'n_rows_per_quality', 'n_families_per_quality']:
            assert abs(actual[field] - expected[field]) < 1e-12
    sql_detail = [dict(r) for r in db.execute(detail_sql)]
    assert len(sql_detail) == len(detail)
    for expected in detail:
        actual = next(r for r in sql_detail if r['quality'] == expected['quality'])
        for field in ['rho_covariance', 'rho_euclidean', 'mae_covariance', 'mae_euclidean', 'rho_delta', 'mae_delta']:
            assert abs(float(actual[field]) - expected[field]) < 1e-12
    for sid, filename, table, sql in [('ridge', 'metrics.csv', 'run_metrics', ridge_sql),
                                       ('detail', 'ridge_comparison.csv', 'run_ridge_comparison', detail_sql)]:
        src = next(s for s in manifest['sources'] if s['id'] == sid)
        src['path'] = 'runs/' + run_name + '/' + filename
        src['query'].update(sql=sql, language='sql', engine='sqlite', tables_used=[table])
        src['query']['description'] += (' The raw source CSV named by this source path is imported without row filtering into SQLite table '
                                        + table + ' by analysis/geometry_v1_html.py. This query recomputes the reviewed values; wording labels and chart sorting are display-only transformations.')

    def md(sid, body, provenance=None):
        block = dict(id=sid, type='markdown', body=body)
        if provenance:
            block['sourceId'] = provenance
        manifest['blocks'].append(block)

    def chart(cid, title, subtitle, rows, x, y, ylabel, source_id, group=None, refs=()):
        snapshot['datasets'][cid] = rows
        enc = dict(x=dict(field=x, type='nominal', label=''),
                   y=dict(field=y, type='quantitative', label=ylabel))
        if group:
            enc['color'] = dict(field=group, type='nominal', label='Method')
        enc['tooltip'] = [dict(field=c, type='quantitative' if isinstance(rows[0][c], (int, float)) else 'nominal',
                               label=c.replace('_', ' ').capitalize()) for c in rows[0] if c not in [x, y, group]]
        manifest['charts'].append(dict(
            id=cid, title=title, subtitle=subtitle, showDescription=True, type='bar',
            dataset=cid, sourceId=source_id, layout='full', encodings=enc,
            settings=dict(orientation='horizontal', groupMode='grouped' if group else 'single',
                          categoryLabelPolicy='wrap', sort='none'),
            labels=dict(values='all'), legend=dict(sort='spec'),
            palette=dict(kind='categorical' if group else 'sequential'), valueFormat='number',
            referenceLines=list(refs)))
        manifest['blocks'].append(dict(id=cid + '_block', type='chart', chartId=cid, layout='full'))

    methods = [('ridge_euclidean', 'Euclidean'), ('ridge_covariance', 'Covariance'),
               ('ridge_v3_scaled_same_site', 'Standardized')]
    rows = []
    for wording, key in [('Familiar wording', 'entity_test|graded|plain'),
                         ('New wording', 'entity_test+template|graded|paraphrase')]:
        for method, label in methods:
            r = next(r for r in controls if r['partition'] == key and r['method'] == method)
            rows.append(dict(wording=wording, method=label, ordering=r['rho'], error=r['mae'],
                             properties=14, examples_per_property=10, independent_families_per_property=2))

    md('title', '# ' + title)
    md('summary', '''## Executive Summary

- **The covariance method did not consistently improve the measured quality representations.** It helped some familiar-wording scores, but average ordering became worse with new wording.
- **A better average can hide widespread setbacks.** New-wording error fell overall, yet increased for 9 of the 14 properties. The standardized control remained strongest in the aggregate comparisons.
- **This is a small, controlled screen.** It shows how different measurement rules behave inside one frozen model; it does not establish universal concept axes or improved model answers.''', 'findings')

    md('design', '''## What we changed—and what the test asks

Could a different way of measuring distances and directions inside a language model reveal cleaner concepts such as size, speed or sweetness?

We reused **about 6.3k authored examples** and the frozen **Qwen3-1.7B-Base** model. We read the internal vector just before its output head turns that vector into scores for possible next tokens. The original examples, splits, runs and caches were preserved.

Think of **covariance geometry as a new ruler**: it adjusts for patterns of variation shared across the model’s output-word vectors. We computed it from the full output vocabulary, about 152k rows. Context vectors and word vectors receive complementary transformations, preserving the model’s original token scores. No model training or steering was performed.

We then fitted simple linear scoring rules, called **ridge probes**, to predict the authored property levels. Three rules use the **same internal location**: **Euclidean** keeps the original coordinate scales; **Covariance** uses the new ruler; **Standardized** rescales each coordinate separately, following the earlier V3 preprocessing. Refitting can change the probe’s predictions because the penalty on its coefficients changes; it does not add information to the frozen model.''', 'findings')

    md('ordering_intro', '''## New wording remains the harder test

**Ordering asks: can the probe put examples in the right sequence?** Spearman’s score is 1 for perfect ordering, 0 for no consistent ordering and −1 for reversed ordering. It is not percentage accuracy. Both conditions below use unseen entity families; “new wording” also uses phrasing withheld from fitting.

The covariance bars rise slightly for familiar wording, but fall for new wording: **0.543 → 0.498**. Standardized is highest in both conditions. Each bar averages 14 properties equally; each property has only **10 examples from two independent test families**.''', 'ridge')
    chart('ordering', 'Ordering scores across wording conditions',
          'Higher is better · mean of 14 properties · 2 test families per property',
          rows, 'wording', 'ordering', 'Ordering score', 'ridge', 'method',
          [dict(axis='x', value=1, label='Perfect order', color='neutral', lineStyle='dashed')])

    md('error_intro', '''**Error asks a different question: how close is the predicted number?** Mean absolute error (MAE) averages the size of prediction mistakes. Lower is better. Labels run from 0 to 1; a 0.25 error is one interval in the designed five-level examples, not a measured human perceptual distance. Predictions can leave that range, so error can exceed 1.

Covariance lowers mean error in both conditions, including **0.425 → 0.356** with new wording. Standardized is still lowest. A probe can put examples in the correct order while assigning numbers that are too high or too spread out.''', 'ridge')
    chart('error', 'Prediction error across wording conditions',
          'Lower is better · authored 0–1 scale · the dashed line is one designed level interval',
          rows, 'wording', 'error', 'Average absolute error', 'ridge', 'method',
          [dict(axis='x', value=.25, label='One level interval', color='neutral', lineStyle='dashed')])

    md('uneven', '''## Most properties did not share the average error gain

**Read left as lower error with covariance, right as higher error.** All 14 properties are shown. Error improves for 5 and worsens for 9; the median change is a small worsening of **+0.005**. Arousal—roughly, how activated or excited something is—has a large correction, **1.556 → 0.669**, which drives much of the average gain.

Speed illustrates the difference between ordering and calibration: its ordering improves slightly, yet its error rises **0.109 → 0.417**. Across properties, ordering improves for 5, worsens for 8 and stays unchanged for 1. The aggregate results therefore do not support a general improvement.''', 'detail')
    delta_rows = [dict(property=r['quality'].capitalize(), error_change=r['mae_delta'],
                       euclidean_error=r['mae_euclidean'], covariance_error=r['mae_covariance'],
                       ordering_change=r['rho_delta'], examples=10, independent_families=2)
                  for r in sorted(detail, key=lambda r: r['mae_delta'])]
    chart('property_changes', 'New-wording error change for each property',
          'Covariance minus Euclidean · negative = improvement · positive = worsening',
          delta_rows, 'property', 'error_change', 'Change in absolute error', 'detail', refs=[
              dict(axis='x', value=0, label='No change', color='neutral', lineStyle='dashed')])

    md('separation', '''## We still do not see clean, independent property directions

**Consistency means that increasing the same property points in a similar internal direction across examples.** Cosine alignment is 1 for the same direction, 0 for perpendicular directions and −1 for opposite directions. With new wording it stays low: **0.045 → 0.044**. Familiar-wording alignment changes little too: **0.176 → 0.178**. The new ruler does not clearly make these changes line up.''', 'consistency')
    md('factorial', '''**Selectivity means responding to the property that changed while leaving other property scores stable.** In 270 matched numeric contrasts from one test family, the intended property’s mean signed response grows **0.049 → 0.058**, but the other properties’ mean absolute movement also grows **0.064 → 0.067**. Reordered wording shows the same tension. These are different signed and absolute statistics, not an accuracy ratio; the result does not demonstrate isolated property axes.''', 'factorial')
    md('binding', '''**Binding means attaching a property to the right object:** “A is large, B is small” should differ from the swapped assignment even though the words are the same. For the simple direction-based readouts, the fraction of changes with the intended sign rises **71.4% → 76.8%** with familiar wording and stays **64.3%** with new wording. Reordering clauses has similar or slightly smaller effects, while irrelevant edits disturb scores slightly more. These are readout checks, not the model’s answer accuracy; binding uses only one test family.''', 'binding')
    md('coverage', '''## Colour and word pairs provide limited extra evidence

**Hue goes around a circle, so we measure angular error.** Smaller angles are better: word-based hue error changes **43.57° → 42.16°**, while numeric hue worsens **69.15° → 70.71°**. These opposing changes, each from two families, do not support a general gain.

**The output-word test asks whether high-minus-low word pairs agree on a direction.** Exact single-token filtering left training pairs for only **4 of 14 properties**, and both training and test pairs for only **sweetness**. Its two test pairs share three concepts; alignment changes **0.100 → 0.096**. Nouns also differ in identity and other properties. This is too little evidence to settle the hypothesis; missing pairs are missing evidence, not failed tests.''', 'findings')

    md('next', '''## What this supports, and what to test next

**Keep covariance as a comparison, rather than treating it as the preferred measurement rule.** The strongest aggregate predictive control here is standardized ridge. The open question is whether covariance helps with broader independent examples and better-controlled word pairs.

Before drawing a general conclusion, expand the independent test families and eligible concept pairs, keep all three controls at the same model location, and judge ordering, numerical error and selectivity separately. Earlier V3 headline results used different readout locations, so they are historical context rather than a matched before-and-after comparison.''', 'findings')
    md('limits', '''## How much confidence to place in this

**The implementation is checked; the scientific conclusion remains limited.** The smoke and full laptop runs completed, 41 offline tests passed, and fresh model checks reproduced the cached states and output scores exactly for the checked prompts. A separate audit recomputed about 3.3k metric cells with zero discrepancy.

The labels are authored examples, not human rating norms. Many contrasts reuse a small number of independent families; they should not be counted as independent replications. There is no significance claim, no causal intervention, and no evidence here that the model acquired new knowledge or more human-like concepts. The result is a useful constraint on the hypothesis: changing this ruler was not enough to produce a consistent improvement.''', 'findings')

    diagrams = add_diagrams(manifest, snapshot, out)
    artifact = dict(surface='report', manifest=manifest, snapshot=snapshot, sources=manifest['sources'])
    (out / 'artifact.json').write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    notes = dict(
        scope='Concise unified reader for the completed covariance extension; earlier V3 findings remain historical context.',
        audience='General research stakeholder; executive-report shape. Design follows summary; next steps and further questions are combined; caveats close the report.',
        chart_contracts=[
            dict(id='ordering', question='Does the ruler help correct ordering transfer to new wording?',
                 family='comparison', type='grouped horizontal bar', fields=['wording', 'ordering', 'method'],
                 claim='New-wording ordering declines under covariance; standardized control has the highest mean.',
                 palette='Native categorical palette, fixed Euclidean/Covariance/Standardized series order; three identities, direct values and legend.'),
            dict(id='error', question='Are numerical scores closer to authored levels?',
                 family='comparison', type='grouped horizontal bar', fields=['wording', 'error', 'method'],
                 claim='Mean error falls under covariance, but standardized has lower error in both conditions.',
                 rationale='Same layout as ordering intentionally supports comparison of two distinct metrics.',
                 palette='Same fixed categorical method order as ordering.'),
            dict(id='property_changes', question='Do most properties share the average gain?',
                 family='distribution/comparison', type='signed horizontal bars', fields=['property', 'error_change'],
                 claim='Five improve and nine worsen; one large arousal correction affects the mean.',
                 palette='One blue root; sign, position about zero, direct values and explicit reading instructions.')],
        omitted_visuals='All four completed covariance scientific figures and the specifically discussed historical PCA map are included. Smoke duplicates and old pilot steering/layer-sweep figures are outside this covariance measurement comparison; their historical source runs remain preserved. Context consistency, binding and hue retain their exact-value explanations. No unsupported uncertainty intervals.',
        reproduction='python analysis/geometry_v1_html.py --output reports/quality_geometry_v1_findings; then use the Data Analytics report:deliver command on artifact.json.',
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                      if p.name in ['ridge_controls.csv', 'heldout_per_quality.csv', 'findings.md',
                                    'context_consistency.csv', 'binding_nuisance_summary.csv', 'matched_factorial_summary.csv']},
        validation=dict(summary_means_recomputed=True, delta_signs_and_counts_checked=True,
                        original_metric_cells_recomputed=3258, original_audit='validation.json',
                        original_data_mutated=False, inference_performed=False))
    notes['chart_contracts'].extend(diagrams['contracts'])
    notes['figure_sha256'] = diagrams['image_sha256']
    notes['validation']['original_blocks_preserved'] = diagrams['original_blocks_preserved']
    notes['validation']['added_figures'] = diagrams['added_figures']
    (out / 'html_report_notes.json').write_text(json.dumps(notes, indent=2) + '\n')
    print(json.dumps(dict(artifact=str(out / 'artifact.json'), charts=len(manifest['charts']),
                          blocks=len(manifest['blocks']), data_rows=sum(map(len, snapshot['datasets'].values())))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
