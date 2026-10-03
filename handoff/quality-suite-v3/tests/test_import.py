import pandas as pd
import pytest
from quality_dimensions.data import import_data,load_data,validate


def test_import_preserves_provenance_and_separate_raters(fixture_config,tmp_path):
    c,r,p=load_data(fixture_config)
    c['label_kind']='human_ordinal'; c['label_source']='TEST FIXTURE ONLY: source metadata'; c['uncertainty']='not estimated';c['aggregation']='median'
    source=tmp_path/'input.csv'; c.to_csv(source,index=False)
    # Explicitly supplied synthetic test observation: not included in research data.
    ratings=tmp_path/'ratings.csv';pd.DataFrame([dict(concept_id=c.concept_id.iloc[0],rater_id='unit_test_fixture',rating=1)]).to_csv(ratings,index=False)
    out=tmp_path/'imported.csv';import_data(source,out,fixture_config['references'],fixture_config['pairs'],ratings)
    assert pd.read_csv(out).uncertainty.eq('not estimated').all()
    assert pd.read_csv(out.with_name('imported_raters.csv')).rater_id.iloc[0]=='unit_test_fixture'
    bad=c.copy();bad['label_kind']='sourced_measurement'
    with pytest.raises(ValueError,match='units'):validate(bad,r,p)
