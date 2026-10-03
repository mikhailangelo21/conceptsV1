"""Add the existing scientific figures requested by the reader to the report.

The native summary charts remain intact. Original figures are preserved byte for
byte in canonical HTML blocks, retaining categorical dots, marker shapes, shared
heatmap scales and the historical PCA's original coordinate system.
"""
import base64
import hashlib
import html
import struct
from pathlib import Path


def add_diagrams(manifest, snapshot, out: Path):
    report_prefix = 'reports/quality_geometry_v1_findings/'
    run_prefix = 'runs/quality_geometry_v1_laptop-e95e9cf10513/'
    pilot_prefix = 'runs/pilot_m2-792030855aa5/'
    contracts = []
    hashes = {}

    def diagram(cid, filename, source_path, label, explanation, alt, source_tables, methods,
                question, takeaway, palette, historical=False):
        image_bytes = (out / filename).read_bytes()
        assert image_bytes[:8] == b'\x89PNG\r\n\x1a\n'
        width, height = struct.unpack('>II', image_bytes[16:24])
        hashes[filename] = hashlib.sha256(image_bytes).hexdigest()
        data_url = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('ascii')
        sid = cid + '_source'
        manifest['sources'].append(dict(
            id=sid, label=label, path=source_path,
            query=dict(engine='saved scientific figure', description=methods,
                       tables_used=source_tables,
                       filters=['Historical size-only pilot; never pooled with the covariance run.'] if historical else
                               ['Completed full-vocabulary covariance laptop run; no smoke data included.'])))
        escaped_alt = html.escape(alt, quote=True)
        figure = (
            '<figure style="margin:0;font:14px/1.5 system-ui,sans-serif;color:CanvasText">'
            f'<img src="{data_url}" alt="{escaped_alt}" width="{width}" height="{height}" '
            'style="display:block;width:100%;height:auto;background:white">'
            '<details style="margin:10px 0 0"><summary style="cursor:pointer">'
            'Enlarge diagram for full-size labels</summary>'
            '<div style="max-width:100%;overflow-x:auto;margin-top:10px" tabindex="0" '
            'role="region" aria-label="Full-size diagram; scroll horizontally">'
            f'<img src="{data_url}" alt="{escaped_alt}" width="{width}" height="{height}" '
            f'style="display:block;width:{width}px;min-width:{width}px;max-width:none;height:auto;background:white">'
            '</div></details></figure>')
        contracts.append(dict(id=cid, question=question, claim=takeaway,
                              family=label, type='preserved scientific figure',
                              original_image=source_path, source_files=source_tables,
                              palette=palette, historical=historical,
                              delivery='Original PNG embedded in the canonical artifact; responsive overview and keyboard-accessible full-resolution expansion. No separate runtime or network assets.',
                              scope='User-requested original diagrams, added without replacing the native summary charts.',
                              dimensions=[width, height]))
        return [dict(id=cid + '_reading', type='markdown', body=explanation, sourceId=sid),
                dict(id=cid, type='html', body=figure, sourceId=sid, layout='full')]

    additions = {}
    additions['property_changes_block'] = diagram(
        'ridge_dot_plot', 'ridge_changes.png', run_prefix + 'plots/ridge_changes.png',
        'Per-property changes in ordering and error',
        '''### Dot plots: where the gains and losses occur

Each row is a property. **Blue circles are familiar wording; orange squares are new wording.** The vertical zero line means covariance made no difference. In the **left panel, right is better** because ordering increased. In the **right panel, left is better** because error decreased. The distance from zero shows the size of the change; it is not an uncertainty interval.

These plots add both wording conditions and both measurements to the error chart above. Improvements are scattered rather than universal. For example, arousal’s familiar-wording ordering improves, while its new-wording ordering worsens. Each point summarizes only two independent test families.''',
        'Two dot-plot panels for 14 properties. Covariance minus Euclidean ordering appears on the left, where positive is better; error change appears on the right, where negative is better. Familiar and held-out wording often disagree.',
        [run_prefix + 'ridge_comparison.csv'],
        'Saved geometry_report.py figure. All 14 properties, identical final readout, entity-test graded plain and paraphrase partitions. Points are covariance-minus-Euclidean changes, with two independent test families each; no confidence intervals.',
        'Which properties improve or worsen, and does that depend on wording?',
        'Familiar and held-out wording differ; ordering and calibration changes are not consistently beneficial.',
        'Blue circles and orange squares; paired row position and explicit zero lines.')

    additions['factorial'] = diagram(
        'factorial_dot_plot', 'factorial_selectivity.png', report_prefix + 'factorial_selectivity.png',
        'Target and off-target sensitivity',
        '''**Read each row as a before-and-after comparison of the measurement rule.** Blue circles use Euclidean geometry and orange squares use covariance. For a **target** row, a larger positive response is desirable: the intended property was increased. For an **off-target** row, smaller is desirable: other measured properties should stay still.

The orange points move right in both kinds of row. Covariance increases the intended response slightly, but also increases movement in the other readouts. It therefore does not cleanly isolate the edited property. The figure uses the same 270 contrasts per wording, from just one test family; target values are signed means and off-target values are absolute means.''',
        'Four paired dot comparisons: familiar numeric and held-out reordered wording, each with target signed response and off-target absolute response. Covariance is slightly higher in all four rows.',
        [report_prefix + 'matched_factorial_summary.csv', report_prefix + 'matched_factorial_changes.csv'],
        'Original analysis/geometry_v1_findings.py figure. Matched contrasts are averaged within independent family. The target statistic is signed; off-target responses are absolute and averaged over other measured scalar properties. One test family and 270 contrasts per wording.',
        'Does covariance strengthen the intended response without disturbing other properties?',
        'Both target and off-target responses increase slightly; clean separation is not established.',
        'Blue circles and orange squares; printed values and zero reference.')

    additions['coverage'] = diagram(
        'word_alignment_dot_plot', 'output_heldout_alignment.png', run_prefix + 'plots/output_heldout_alignment.png',
        'Alignment of held-out output-word pairs',
        '''### Dot plot: do unseen word pairs agree with the learned direction?

The circle and square mark the mean alignment of the two eligible sweetness test pairs. **Zero means perpendicular, +1 means the same direction and −1 means the opposite direction.** The crosses show the mean for 512 random output-row pairs under each rule, not confidence bounds.

Both test averages sit slightly to the right of zero, at about 0.10, and almost overlap across geometries. This is weak positive alignment, not a large covariance gain. The two test pairs share three concepts; their small, dependent sample cannot establish a general result. Other properties are absent because they lack both eligible training and test pairs.''',
        'Sweetness held-out alignment: Euclidean mean cosine about 0.100, covariance about 0.096, with random-pair means near zero. Two test pairs share three concepts.',
        [run_prefix + 'output_alignment_summary.csv', run_prefix + 'output_pair_counts.csv'],
        'Original geometry_report.py output-word figure. Original test-split pairs only; random comparison uses 512 uniformly sampled full-output-head row pairs per quality. Only sweetness has an eligible training direction and test pairs.',
        'Do held-out output-word differences point along the training direction?',
        'The two geometries give almost the same weak positive alignment; coverage is severely limited.',
        'Blue circle, orange square and matching random-pair crosses; common −1 to +1 scale.')
    additions['coverage'] += diagram(
        'direction_heatmaps', 'output_direction_similarity.png', run_prefix + 'plots/output_direction_similarity.png',
        'Similarity between training output-word directions',
        '''### Heatmaps: do different properties point in different directions?

Choose a row and column to compare two property directions. **Darker squares mean more aligned; lighter squares mean closer to perpendicular.** The score is absolute cosine: 0 means perpendicular, while 1 means parallel in either orientation. The dark diagonal compares each direction with itself, so it is expected.

The two panels use the same colour scale and look very similar. Mass and size have the largest off-diagonal similarity, about 0.17, under both rules. Most other pairs are closer to perpendicular. But these are only four available **training** directions: a pale square does not prove that a property is independent, causally isolated or transferable to new examples.''',
        'Two 4-by-4 absolute-cosine heatmaps for mass, roughness, size and sweetness. Euclidean and covariance patterns are very similar. Diagonal values are one; the mass-size comparison is about 0.17.',
        [run_prefix + 'output_direction_similarity.csv', run_prefix + 'output_pair_counts.csv'],
        'Saved geometry_report.py heatmaps, using mean training-pair directions. Equal 0–1 colour scales, printed values. Only four properties have single-token training pairs. No causal or dimensionality interpretation follows from near-perpendicularity.',
        'Does the covariance metric make different property directions more distinct?',
        'The four available training directions retain a very similar similarity pattern.',
        'One blue sequential root, shared scale and printed cell values.')

    # This historical figure remains separate from the current experiment.
    historical = diagram(
        'historical_pca_map', 'pilot_pca_selected.png', pilot_prefix + 'plots/pca_selected.png',
        'Earlier pilot: two-dimensional concept map',
        '''## Historical context: the earlier 2D concept map

**This is the earlier size-only pilot at block 23, not a covariance-versus-Euclidean comparison.** Each mark represents a concept: training marks average its neutral prompts, while test marks use the held-out neutral wording. Shape identifies the object category. Colour shows the provisional size rank, from dark for smaller to yellow for larger; larger outlined marks are test examples.

PCA makes a flat map by retaining the two directions with the most variation in the training vectors, then placing test examples on those same axes. Nearby points are similar **in this projection**. Neither axis is defined as “size.” Together they show only about **21.3% of training variation**, so distance and separation here leave out most of the original space.

Use this map to explore patterns, not to prove a clean size axis. Category and other features may also shape the picture. Its provisional labels, earlier representation site and different setup mean it cannot be pooled with the current comparison.''',
        'Historical block-23 PCA map. Colour indicates provisional ordinal size; shapes indicate six object categories; larger outlined marks are test concepts. PC1 explains 11.5 percent and PC2 9.8 percent of training variance.',
        [pilot_prefix + 'pca.csv', pilot_prefix + 'selection.json', pilot_prefix + 'config.json'],
        'Preserved pilot figure from src/quality_dimensions/report.py, not newly fitted. Training: neutral rows from seen templates, averaged by concept_id, category, size_score and split. Test: neutral rows from held-out template. Fixed train-only PCA at validation-selected block 23. Axis variance shares 11.5% and 9.8% are read from the saved figure. Local copy pilot_pca_selected.png is byte-identical to the original.',
        'What did the earlier exploratory map show, and how should it be read?',
        'The map is a low-dimensional exploratory view, not a test of the covariance extension or proof of a semantic axis.',
        'Preserved size colour scale, category shapes, and outlined larger test marks.', historical=True)

    old_blocks = list(manifest['blocks'])
    updated = []
    for block in old_blocks:
        if block['id'] == 'next':
            updated.extend(historical)
        updated.append(block)
        updated.extend(additions.get(block['id'], []))
    assert len(updated) == len(old_blocks) + 10
    original_ids = {b['id'] for b in old_blocks}
    assert [b for b in updated if b['id'] in original_ids] == old_blocks
    manifest['blocks'] = updated
    return dict(contracts=contracts, image_sha256=hashes,
                original_blocks_preserved=True, added_figures=5)
