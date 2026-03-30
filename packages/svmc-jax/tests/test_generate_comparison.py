import importlib.util
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "generate_comparison.py"


def _load_generate_comparison_module():
    spec = importlib.util.spec_from_file_location("generate_comparison", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_series(keys, value):
    return {key: value for key in keys}


def test_build_comparison_rejects_missing_fortran_day():
    module = _load_generate_comparison_module()
    ref = {"daily": {"dates": ["d1", "d2"], "lai_day": [0.1, 0.2]}}
    keys = module.COMPARE_KEYS

    fortran_records = [
        {"inputs": {"day": 1}, "output": _make_series(keys, 1.0)},
    ]
    jax_records = [
        {"day": 1, **_make_series(keys, 1.0), "et_total": 0.1},
        {"day": 2, **_make_series(keys, 1.0), "et_total": 0.2},
    ]

    with pytest.raises(RuntimeError, match="missing days"):
        module.build_comparison(ref, fortran_records, jax_records, ndays=2)


def test_build_metadata_includes_provenance():
    module = _load_generate_comparison_module()
    meta = module.build_metadata(module.QVIDJA_REF, ndays=3)

    assert meta["git_commit"]
    assert isinstance(meta["git_dirty"], bool)
    assert len(meta["qvidja_reference_sha256"]) == 64
    assert meta["generated_at_utc"].endswith("Z")
