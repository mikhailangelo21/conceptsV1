import numpy as np
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

@pytest.fixture
def tiny():
    torch.manual_seed(7)
    config=Qwen3Config(vocab_size=64,hidden_size=24,intermediate_size=48,num_hidden_layers=3,num_attention_heads=3,num_key_value_heads=1,head_dim=8,attention_dropout=0.,use_cache=False)
    config._attn_implementation='eager'
    return Qwen3ForCausalLM(config).eval().requires_grad_(False)

@pytest.fixture
def fixture_config():
    from quality_dimensions.pipeline import configuration
    from pathlib import Path
    return configuration(Path(__file__).parents[1]/'configs'/'pilot_m2.yaml')
