#!/usr/bin/env python3
"""Generate Fortran vs JAX comparison data for the Qvidja reference dataset.

Runs both the Fortran SVMC harness and the JAX integration model for
the full Qvidja reference dataset (1697 days), then writes a comparison
JSON suitable for the website.

Usage:
    python scripts/generate_comparison.py [--ndays N]

Outputs:
    website/public/comparison-qvidja.json
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from svmc_jax.qvidja_replay import build_qvidja_run_kwargs

REPO_ROOT = Path(__file__).resolve().parent.parent
SVMC_REF = REPO_ROOT / "packages" / "svmc-ref"
QVIDJA_REF = REPO_ROOT / "website" / "public" / "qvidja-v1-reference.json"
OUTPUT_PATH = REPO_ROOT / "website" / "public" / "comparison-qvidja.json"
TMP_WORK_ROOT = REPO_ROOT / "tmp" / "comparison-export"

COMPARE_KEYS = [
    "gpp_avg", "nee", "hetero_resp", "auto_resp",
    "cleaf", "croot", "cstem", "cgrain",
    "lai_alloc", "litter_cleaf", "litter_croot",
    "soc_total", "wliq", "psi",
]

# ── Fortran side ──────────────────────────────────────────────────────


def prepare_fortran_workspace() -> Path:
    """Create an isolated temporary Fortran workspace under tmp/."""
    if TMP_WORK_ROOT.exists():
        shutil.rmtree(TMP_WORK_ROOT)

    work_root = TMP_WORK_ROOT / "repo"
    work_svmc_ref = work_root / "packages" / "svmc-ref"
    work_vendor_src = work_root / "vendor" / "SVMC" / "src"
    work_qvidja_ref = work_root / "website" / "public" / "qvidja-v1-reference.json"

    shutil.copytree(
        SVMC_REF,
        work_svmc_ref,
        ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc", "fixtures.jsonl"),
    )
    shutil.copytree(REPO_ROOT / "vendor" / "SVMC" / "src", work_vendor_src)
    work_qvidja_ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(QVIDJA_REF, work_qvidja_ref)
    return work_svmc_ref


def load_generate_module(work_svmc_ref: Path):
    """Import the temp-workspace generate.py as a one-off module."""
    module_path = work_svmc_ref / "generate.py"
    spec = importlib.util.spec_from_file_location("svmc_ref_generate_temp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fortran(ndays: int) -> list[dict]:
    """Build and run the Fortran harness for *ndays* days in tmp/.

    This keeps website export generation from mutating tracked fixture or
    build state in packages/svmc-ref.
    """
    work_svmc_ref = prepare_fortran_workspace()
    gen = load_generate_module(work_svmc_ref)

    # Patch the module-level constants
    gen.INTEGRATION_NDAYS = ndays
    gen.INTEGRATION_NHOURS = ndays * 24

    # Clean temp fpm build cache to force recompilation with new include files
    fpm_build_dir = work_svmc_ref / "build"
    if fpm_build_dir.exists():
        shutil.rmtree(fpm_build_dir)

    print(f"[Fortran] Building harness for {ndays} days …")
    gen.build_harness()

    print(f"[Fortran] Running harness …")
    records = gen.run_harness()

    # Filter for integration_daily records only
    integration = [r for r in records if r["fn"] == "integration_daily"]
    print(f"[Fortran] Got {len(integration)} integration_daily records")
    return integration


# ── JAX side ──────────────────────────────────────────────────────────


def run_jax(ref: dict, ndays: int) -> list[dict]:
    """Run the JAX integration model for *ndays* days."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from svmc_jax.integration import run_integration

    print(f"[JAX] Running integration for {ndays} days …")
    t0 = time.time()
    _final_carry, daily_outputs = run_integration(**build_qvidja_run_kwargs(ref, ndays))
    elapsed = time.time() - t0
    print(f"[JAX] Done in {elapsed:.1f}s")

    # Extract daily outputs as plain Python lists
    scalar_keys = [
        "gpp_avg", "nee", "hetero_resp", "auto_resp",
        "cleaf", "croot", "cstem", "cgrain",
        "lai_alloc", "litter_cleaf", "litter_croot",
        "soc_total", "wliq", "psi", "et_total",
    ]

    jax_records = []
    for day_idx in range(ndays):
        out = {"day": day_idx + 1}
        for key in scalar_keys:
            out[key] = float(getattr(daily_outputs, key)[day_idx])
        out["cstate"] = [float(x) for x in daily_outputs.cstate[day_idx]]
        jax_records.append(out)

    print(f"[JAX] Extracted {len(jax_records)} daily records")
    return jax_records


# ── Combine & write ──────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def get_git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(status.strip())


def build_metadata(ref_path: Path, ndays: int) -> dict:
    return {
        "site": "Qvidja",
        "ndays": ndays,
        "generated_by": "scripts/generate_comparison.py",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "qvidja_reference_sha256": sha256_file(ref_path),
        "compare_keys": COMPARE_KEYS,
    }


def index_records_by_day(records: list[dict], *, label: str, ndays: int, day_getter) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for record in records:
        day = int(day_getter(record))
        if day < 1 or day > ndays:
            raise RuntimeError(f"{label} record day {day} outside expected range 1..{ndays}")
        if day in indexed:
            raise RuntimeError(f"{label} records contain duplicate day {day}")
        indexed[day] = record

    missing_days = [day for day in range(1, ndays + 1) if day not in indexed]
    if missing_days:
        raise RuntimeError(f"{label} records missing days: {missing_days[:10]}")
    return indexed


def build_comparison(
    ref: dict,
    fortran_records: list[dict],
    jax_records: list[dict],
    ndays: int,
) -> dict:
    """Build the comparison JSON structure."""
    daily = ref["daily"]
    dates = daily.get("dates", [f"day-{i+1}" for i in range(ndays)])[:ndays]

    fortran_by_day = index_records_by_day(
        fortran_records,
        label="Fortran",
        ndays=ndays,
        day_getter=lambda record: record["inputs"]["day"],
    )
    jax_by_day = index_records_by_day(
        jax_records,
        label="JAX",
        ndays=ndays,
        day_getter=lambda record: record["day"],
    )
    ordered_days = list(range(1, ndays + 1))

    fortran_series = {}
    jax_series = {}
    summary = {}

    import numpy as np

    for key in COMPARE_KEYS:
        f_vals = [fortran_by_day[day]["output"][key] for day in ordered_days]
        j_vals = [jax_by_day[day][key] for day in ordered_days]
        fortran_series[key] = f_vals
        jax_series[key] = j_vals

        errs = np.array([j - f for j, f in zip(j_vals, f_vals)])
        f_vals_np = np.array(f_vals)
        nonzero = np.abs(f_vals_np) > 1e-15
        rel_errs = np.zeros_like(errs)
        np.divide(np.abs(errs), np.abs(f_vals_np), out=rel_errs, where=nonzero)
        summary[key] = {
            "max_abs_error": float(np.max(np.abs(errs))),
            "mean_abs_error": float(np.mean(np.abs(errs))),
            "max_rel_error": float(np.max(rel_errs)),
            "mean_rel_error": float(np.mean(rel_errs[nonzero])) if np.any(nonzero) else 0.0,
            "r_squared": float(np.corrcoef(f_vals, j_vals)[0, 1] ** 2)
                if np.std(f_vals) > 1e-15 else 1.0,
        }

    # JAX-only fields (not in Fortran harness output)
    jax_series["et_total"] = [jax_by_day[day]["et_total"] for day in ordered_days]

    # LAI from reference for context
    lai_series = [float(x) for x in daily["lai_day"][:ndays]]

    return {
        "meta": {
            **build_metadata(QVIDJA_REF, ndays),
            "dates": dates,
        },
        "lai": lai_series,
        "fortran": fortran_series,
        "jax": jax_series,
        "summary": summary,
    }


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ndays", type=int, default=None,
        help="Number of days to run (default: all available)",
    )
    args = parser.parse_args()

    print("Loading Qvidja reference data …")
    with open(QVIDJA_REF) as f:
        ref = json.load(f)

    max_days = len(ref["daily"]["lai_day"])
    ndays = args.ndays or max_days
    ndays = min(ndays, max_days)
    print(f"Running comparison for {ndays} days (max available: {max_days})")

    fortran_records = run_fortran(ndays)
    jax_records = run_jax(ref, ndays)

    comparison = build_comparison(ref, fortran_records, jax_records, ndays)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
