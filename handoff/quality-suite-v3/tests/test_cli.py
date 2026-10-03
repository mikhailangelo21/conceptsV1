import json
import subprocess
import sys


def test_clean_process_cli_doctor():
    # A clean process catches import/entry-point failures hidden by pytest's imports.
    result=subprocess.run([sys.executable,'-m','quality_dimensions.cli','doctor'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    data=json.loads(result.stdout)
    assert 'mps_available' in data and data['versions']['transformers']
