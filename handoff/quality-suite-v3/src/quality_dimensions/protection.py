"""Read-only preservation of original pilot evidence."""
from pathlib import Path
import hashlib
from .util import read_json

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = ROOT / 'runs' / 'pilot_m2-792030855aa5'


def forbid_original_write(path):
    path=Path(path).resolve()
    if path==PROTECTED or PROTECTED in path.parents:
        raise ValueError('The original pilot is immutable. Use qd audit or a separately identified v2 run.')


def evidence_hashes(source):
    source=Path(source).resolve()
    files=[p for p in source.rglob('*') if p.is_file()]
    index=source/'activation_index.json'
    if index.exists(): files += [Path(r['cache_file']) for r in read_json(index)]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def verify_protected():
    manifest=ROOT/'docs/provenance/original_pilot_sha256.json'
    expected=read_json(manifest)
    changed=[p for p,h in expected.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    if changed: raise ValueError(f'Protected pilot files changed: {changed[:5]}')
    return len(expected)
