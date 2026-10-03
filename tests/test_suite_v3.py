import json
from pathlib import Path

import numpy as np
import pytest

from quality_dimensions.suite_analysis import _circular_score, _eligible_fit, _ridge_cv
from quality_dimensions.suite_cache import SuiteActivationCache
from quality_dimensions.suite_data import load_suite, suite_plan, tokenize_case


@pytest.fixture(scope="module")
def suite():
    return load_suite("configs/quality_suite_v3.json", "smoke")


def test_profile_joins_and_independent_counts(suite):
    assert len(suite["cases"]) == 260
    assert len(suite["questions"]) == 48
    assert all(row["low_case_id"] != row["high_case_id"] for row in suite["contrasts"])
    plan = suite_plan(suite)
    assert plan["activation_requests_before_deduplication"] == 520
    assert plan["independent_counts"]["activation_entity_groups"] < len(suite["cases"])


def test_token_span_selects_final_overlapping_subtoken():
    class Tokenizer:
        def __call__(self, text, **_):
            return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 4), (5, 7), (7, 9)]}
    case = {"case_id": "x", "text": "xxxx Kalo", "readout_char_span": [5, 9], "readout_entity": "Kalo"}
    rows = tokenize_case(Tokenizer(), case)
    assert rows[0]["readout_index"] == 2
    assert rows[1]["readout_index"] == 2


def test_fit_filter_excludes_numeric_and_factorial_holdout():
    base = {"case_type": "graded", "entity_split": "train", "template_split": "train", "combination_split": "train"}
    assert _eligible_fit(base)
    assert not _eligible_fit({**base, "case_type": "numeric"})
    assert not _eligible_fit({**base, "case_type": "factorial", "combination_split": "test"})
    assert not _eligible_fit({**base, "entity_split": "validation"})


def test_hue_wrap_and_multioutput_grouped_ridge():
    truth = np.array([[np.sin(np.deg2rad(359)), np.cos(np.deg2rad(359))]])
    pred = np.array([[np.sin(np.deg2rad(1)), np.cos(np.deg2rad(1))]])
    assert _circular_score(truth, pred)["circular_mae_degrees"] == pytest.approx(2)
    x = np.arange(24, dtype=float).reshape(8, 3)
    y = np.column_stack([x[:, 0], -x[:, 1]])
    model, alpha = _ridge_cv(x, y, [0, 0, 1, 1, 2, 2, 3, 3], [.1, 1])
    assert model.predict(x).shape == (8, 2)
    assert alpha in {.1, 1}


def test_cache_reuse_requires_exact_tokens_position_and_identity(tmp_path):
    identity = {"block_numbers": [4, 8], "hidden_size": 3, "model_revision": "abc", "dtype": "float16"}
    cache = SuiteActivationCache(tmp_path, identity)
    row = {"token_ids": [1, 2], "readout_index": 1, "readout": "final"}
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    cache.put(row, value)
    np.testing.assert_array_equal(cache.get(row), value)
    assert cache.get({**row, "readout_index": 0}) is None
    assert SuiteActivationCache(tmp_path, {**identity, "model_revision": "def"}).root != cache.root


def test_profile_preserves_hue_pair_and_factorial_isolation(suite):
    hue = [r for r in suite["cases"] if "hue_sin" in r["targets"]]
    assert hue and all("hue_cos" in r["targets"] for r in hue)
    factorial = [r for r in suite["cases"] if r["case_type"] == "factorial"]
    assert factorial
    for row in factorial:
        if "volume" in row.get("derived_values", {}):
            assert "volume" not in row["primitive_axes"]


def test_teacher_forcing_matches_token_by_token(tiny):
    import torch
    from quality_dimensions.model import Runner
    from quality_dimensions.scoring import score_choices
    runner = Runner(tiny, dict(backend='cpu', num_hidden_layers=3), 1)
    row = dict(token_ids=[1,2,3], readout_index=2, candidate_ids={'0':[8,9,10], '1':[11,12]})
    actual = score_choices(runner, row)
    with torch.inference_mode():
        for label, tail in row['candidate_ids'].items():
            prefix = list(row['token_ids']); expected = 0.
            for token in tail:
                expected += float(tiny(torch.tensor([prefix]),use_cache=False).logits[0,-1].log_softmax(-1)[token])
                prefix.append(token)
            assert actual['log_probs'][label] == pytest.approx(expected, abs=2e-6)
    assert not actual['single_token']


def test_reduced_rank_output_bound_and_prediction(tiny):
    from quality_dimensions.suite_diagnostics import reduced_rank_prediction
    rng=np.random.default_rng(4); x=rng.normal(size=(12,8)); y=x[:,:2]
    assert reduced_rank_prediction(x,y,x,1,2).shape == y.shape
    with pytest.raises(ValueError,match='rank'):
        reduced_rank_prediction(x,y,x,1,3)


def test_validation_and_test_are_separate_and_lexical_not_pooled():
    from quality_dimensions.suite_analysis import _partition, _target_rows
    base=dict(case_type='graded', entity_split='validation', template_split='train', combination_split='train', template_id='plain', targets={'size':0.5})
    assert _partition(base) != _partition({**base, 'entity_split':'test'})
    grouped=_target_rows([base,{**base,'case_type':'lexical'}])
    assert len(grouped['size']) == len(grouped['lexical__size']) == 1


def test_shared_readout_forward_matches_independent_captures(tiny):
    from quality_dimensions.model import Runner
    runner=Runner(tiny,dict(backend='cpu',num_hidden_layers=3),1)
    rows=[dict(token_ids=[1,2,3,4],readout_index=p) for p in [1,3]]
    actual=runner.extract_readouts(rows,[1,3])
    assert len(runner.timings)==1
    for row,value in zip(rows,actual):
        np.testing.assert_allclose(value,runner.extract_blocks(row,[1,3]),rtol=0,atol=1e-6)


def test_test_labels_cannot_change_selected_linear_models(suite,tmp_path):
    import copy
    from quality_dimensions.suite_analysis import analyze_suite
    data=copy.deepcopy(suite)
    data['config']['extraction_block_numbers']=[4]
    data['config']['ridge_alphas']=[.1,1]
    data['config']['subspace_ranks']=[1,2]
    data['token_counts']={r['case_id']:20 for r in data['cases']}
    rng=np.random.default_rng(12)
    activation={(r['case_id'],ro):rng.normal(size=(1,8)).astype(np.float32) for r in data['cases'] for ro in ['entity','final']}
    static={r['case_id']:rng.normal(size=8) for r in data['cases']}
    first=tmp_path/'first'; second=tmp_path/'second'; first.mkdir(); second.mkdir()
    analyze_suite(first,data,activation,static)
    for r in data['cases']:
        if r['entity_split']=='test':
            r['targets']={k:-100-v for k,v in r['targets'].items()}
    analyze_suite(second,data,activation,static)
    assert json.loads((first/'model_selection.json').read_text()) == json.loads((second/'model_selection.json').read_text())
    for p in (first/'fits').glob('*.npz'):
        with np.load(p) as a,np.load(second/'fits'/p.name) as b:
            for key in a.files: np.testing.assert_array_equal(a[key],b[key])
