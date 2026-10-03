import os
from pathlib import Path
import pytest
from quality_dimensions.pipeline import configuration,execute
from quality_dimensions.util import read_json


@pytest.mark.integration
def test_real_model_smoke(tmp_path):
    if os.environ.get('QD_REAL_MODEL')!='1': pytest.skip('Set QD_REAL_MODEL=1 explicitly to load pretrained public weights')
    config=configuration(Path(__file__).parents[1]/'configs'/'smoke.yaml')
    run_path=Path(os.environ.get('QD_REAL_MODEL_RUN_DIR',str(tmp_path/'real_smoke')))
    run=execute(config,run_dir=run_path,resume=True,max_minutes=60)
    state=read_json(run/'state.json')
    assert state['status']=='complete',state
    assert read_json(run/'model.json')['model_revision']==read_json(run/'model.json')['tokenizer_revision']
    assert (run/'report.md').exists() and (run/'plots'/'steering_dose_response.png').exists()

    import pandas as pd
    import numpy as np
    scores=pd.read_csv(run/'behavior_scores.csv')
    assert np.isfinite(scores[['logp_A','logp_B']].to_numpy()).all()
    nonzero=scores[scores.direction.eq('semantic') & scores.strength.ne(0)]
    assert len(nonzero)>0 and nonzero.actual_delta_norm.gt(0).all()
