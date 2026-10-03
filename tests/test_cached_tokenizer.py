import pytest
from transformers import AutoTokenizer
from quality_dimensions.data import load_data,split_concepts,select_subjects
from quality_dimensions.prompts import geometry_prompts,behavioral_prompts,tokenize_prompt,answer_encoding


def test_actual_cached_tokenizer_boundaries(fixture_config):
    try:
        tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base',revision='ea980cb0a6c2ae4b936e82123acc929f1cec04c1',local_files_only=True)
    except OSError: pytest.skip('Public tokenizer not already cached; offline test does not download')
    c,r,p=load_data(fixture_config); splits=split_concepts(c,1729); subjects=select_subjects(c,splits,6)
    rows=geometry_prompts(c,splits,fixture_config); b=behavioral_prompts(c,r,p,splits,subjects,fixture_config)
    geometry=[tokenize_prompt(tok,row,256) for row in rows]
    behavior=[tokenize_prompt(tok,row,256) for row in b]
    assert len(geometry)==780 and len(behavior)==192
    enc=answer_encoding(tok,behavior)
    for row in behavior:
        for label in 'AB':
            assert tok.encode(row['text']+enc['prefix']+label,add_special_tokens=False)==row['token_ids']+[enc['token_ids'][label]]
