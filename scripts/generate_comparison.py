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
import importlib.util
import json
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SVMC_REF = REPO_ROOT / "packages" / "svmc-ref"
QVIDJA_REF = REPO_ROOT / "website" / "public" / "qvidja-v1-reference.json"
OUTPUT_PATH = REPO_ROOT / "website" / "public" / "comparison-qvidja.json"
TMP_WORK_ROOT = REPO_ROOT / "tmp" / "comparison-export"

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
    import jax.numpy as jnp
    from svmc_jax.integration import run_integration

    defaults = ref["defaults"]
    hourly = ref["hourly"]
    daily = ref["daily"]

    nhours = ndays * 24

    # Reshape hourly forcing: (nhours,) → (ndays, 24)
    hourly_temp = jnp.array(hourly["temp_hr"][:nhours]).reshape(ndays, 24)
    hourly_rg = jnp.array(hourly["rg_hr"][:nhours]).reshape(ndays, 24)
    hourly_prec = jnp.array(hourly["prec_hr"][:nhours]).reshape(ndays, 24)
    hourly_vpd = jnp.array(hourly["vpd_hr"][:nhours]).reshape(ndays, 24)
    hourly_pres = jnp.array(hourly["pres_hr"][:nhours]).reshape(ndays, 24)
    hourly_co2 = jnp.array(hourly["co2_hr"][:nhours]).reshape(ndays, 24)
    hourly_wind = jnp.array(hourly["wind_hr"][:nhours]).reshape(ndays, 24)

    daily_lai = jnp.array(daily["lai_day"][:ndays])
    daily_manage_type = jnp.array(
        [float(x) for x in daily["manage_type"][:ndays]]
    )
    daily_manage_c_in = jnp.array(daily["manage_c_in"][:ndays])
    daily_manage_c_out = jnp.array(daily["manage_c_out"][:ndays])

    # Yasso20 MAP parameter vector (from initialize_totc fixture)
    yasso_param = jnp.array([
        0.51, 5.19, 0.13, 0.1, 0.5, 0.0, 1.0, 1.0, 0.99, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.163, 0.0, -0.0, 0.0, 0.0, 0.0,
        0.0, 0.158, -0.002, 0.17, -0.005, 0.067, -0.0, -1.44,
        -2.0, -6.9, 0.0042, 0.0015, -2.55, 1.24, 0.25,
    ])

    print(f"[JAX] Running integration for {ndays} days …")
    t0 = time.time()
    _final_carry, daily_outputs = run_integration(
        hourly_temp=hourly_temp,
        hourly_rg=hourly_rg,
        hourly_prec=hourly_prec,
        hourly_vpd=hourly_vpd,
        hourly_pres=hourly_pres,
        hourly_co2=hourly_co2,
        hourly_wind=hourly_wind,
        daily_lai=daily_lai,
        daily_manage_type=daily_manage_type,
        daily_manage_c_in=daily_manage_c_in,
        daily_manage_c_out=daily_manage_c_out,
        # P-Hydro parameters (Qvidja defaults)
        conductivity=defaults["conductivity"],
        psi50=defaults["psi50"],
        b_param=defaults["b"],
        alpha_cost=defaults["alpha"],
        gamma_cost=defaults["gamma"],
        rdark=defaults["rdark"],
        # SpaFHy soil parameters
        soil_depth=defaults["soil_depth"],
        max_poros=defaults["max_poros"],
        fc=defaults["fc"],
        wp=defaults["wp"],
        ksat=defaults["ksat"],
        n_van=1.14,
        watres=0.0,
        alpha_van=5.92,
        watsat=defaults["max_poros"],
        maxpond=0.0,
        # SpaFHy canopy/aero parameters
        wmax=0.5,
        wmaxsnow=4.5,
        kmelt=2.8934e-5,
        kfreeze=5.79e-6,
        frac_snowliq=0.05,
        gsoil=5.0e-3,
        hc=0.6,
        w_leaf=0.01,
        rw=0.20,
        rwmin=0.02,
        zmeas=2.0,
        zground=0.1,
        zo_ground=0.01,
        # Allocation parameters
        cratio_resp=defaults["cratio_resp"],
        cratio_leaf=defaults["cratio_leaf"],
        cratio_root=defaults["cratio_root"],
        cratio_biomass=defaults["cratio_biomass"],
        harvest_index=defaults["harvest_index"],
        turnover_cleaf=defaults["turnover_cleaf"],
        turnover_croot=defaults["turnover_croot"],
        sla=defaults["sla"],
        q10=defaults["q10"],
        invert_option=defaults["invert_option"],
        pft_is_oat=0.0,
        # Yasso initialization
        yasso_param=yasso_param,
        yasso_totc=defaults["yasso_totc"],
        yasso_cn_input=defaults["yasso_cn_input"],
        yasso_fract_root=defaults["yasso_fract_root"],
        yasso_fract_legacy=0.0,
        yasso_tempr_c=5.4,
        yasso_precip_day=1.87,
        yasso_tempr_ampl=20.0,
    )
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
        out = {}
        for key in scalar_keys:
            out[key] = float(getattr(daily_outputs, key)[day_idx])
        out["cstate"] = [float(x) for x in daily_outputs.cstate[day_idx]]
        jax_records.append(out)

    print(f"[JAX] Extracted {len(jax_records)} daily records")
    return jax_records


# ── Combine & write ──────────────────────────────────────────────────


def build_comparison(
    ref: dict,
    fortran_records: list[dict],
    jax_records: list[dict],
    ndays: int,
) -> dict:
    """Build the comparison JSON structure."""
    daily = ref["daily"]
    dates = daily.get("dates", [f"day-{i+1}" for i in range(ndays)])[:ndays]

    # Fields to compare (present in both Fortran and JAX outputs)
    compare_keys = [
        "gpp_avg", "nee", "hetero_resp", "auto_resp",
        "cleaf", "croot", "cstem", "cgrain",
        "lai_alloc", "litter_cleaf", "litter_croot",
        "soc_total", "wliq", "psi",
    ]

    fortran_series = {}
    jax_series = {}
    summary = {}

    import numpy as np

    for key in compare_keys:
        f_vals = [r["output"][key] for r in fortran_records]
        j_vals = [r[key] for r in jax_records]
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
    jax_series["et_total"] = [r["et_total"] for r in jax_records]

    # LAI from reference for context
    lai_series = [float(x) for x in daily["lai_day"][:ndays]]

    return {
        "meta": {
            "site": "Qvidja",
            "ndays": ndays,
            "dates": dates,
            "generated_by": "scripts/generate_comparison.py",
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
