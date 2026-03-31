#!/usr/bin/env python3
"""Collect normalized Fortran and JAX hourly traces for a Qvidja date window.

The output is written under tmp/ so collection stays separate from analysis.
Fortran records come from the runtime-gated integration_hourly harness log,
while the JAX records are replayed in Python using the same hourly formulas as
the integration loop.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import generate_comparison as comparison
from svmc_jax.integration import (
    _ALPHA_SMOOTH1, _ALPHA_SMOOTH2, _K_EXT, _LAI_GUARD,
    initialization_spafhy, run_integration,
)
from svmc_jax.phydro.leaf_functions import density_h2o
from svmc_jax.phydro.solver import pmodel_hydraulics_numerical
from svmc_jax.qvidja_replay import build_qvidja_run_kwargs
from svmc_jax.water.canopy_soil import CanopySnowParams, canopy_water_flux, soil_water
from svmc_jax.water.leaf_functions import AeroParams, SoilHydroParams


REPO_ROOT = Path(__file__).resolve().parent.parent
QVIDJA_REF = REPO_ROOT / "website" / "public" / "qvidja-v1-reference.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "hourly-logs"
SECONDS_PER_HOUR = 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="Inclusive daily date, e.g. 2022-01-14")
    parser.add_argument("--end-date", required=True, help="Inclusive daily date, e.g. 2022-01-17")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metadata and JSONL outputs. Defaults to tmp/hourly-logs/<start>_to_<end>",
    )
    return parser.parse_args()


def load_reference() -> dict[str, Any]:
    return json.loads(QVIDJA_REF.read_text())


def resolve_day_window(ref: dict[str, Any], start_date: str, end_date: str) -> tuple[int, int]:
    dates = ref["daily"]["dates"]
    try:
        start_day = dates.index(start_date) + 1
    except ValueError as exc:
        raise SystemExit(f"Unknown start date {start_date!r}") from exc
    try:
        end_day = dates.index(end_date) + 1
    except ValueError as exc:
        raise SystemExit(f"Unknown end date {end_date!r}") from exc

    if end_day < start_day:
        raise SystemExit("end date must be on or after start date")
    return start_day, end_day


def scalar(value: Any) -> float:
    return float(jax.device_get(value))


def default_output_dir(start_date: str, end_date: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{start_date}_to_{end_date}"


def attach_timestamp(record: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    inputs = dict(record["inputs"])
    hour_index = int(inputs["hour_index"])
    day = int(inputs["day"])
    inputs["timestamp"] = ref["hourly"]["timestamps"][hour_index - 1]
    inputs["date"] = ref["daily"]["dates"][day - 1]
    return {"fn": record["fn"], "inputs": inputs, "output": record["output"]}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def collect_fortran_hourly(ref: dict[str, Any], start_day: int, end_day: int) -> list[dict[str, Any]]:
    work_svmc_ref = comparison.prepare_fortran_workspace()
    try:
        gen = comparison.load_generate_module(work_svmc_ref)
        gen.INTEGRATION_NDAYS = end_day
        gen.INTEGRATION_NHOURS = end_day * 24
        build_dir = work_svmc_ref / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        gen.build_harness()
        records = gen.run_harness({
            "SVMC_REF_LOG_INTEGRATION_HOURLY": "1",
            "SVMC_REF_LOG_DAY_START": str(start_day),
            "SVMC_REF_LOG_DAY_END": str(end_day),
        })
    finally:
        if comparison.TMP_WORK_ROOT.exists():
            shutil.rmtree(comparison.TMP_WORK_ROOT)

    hourly = [attach_timestamp(record, ref) for record in records if record["fn"] == "integration_hourly"]
    expected = (end_day - start_day + 1) * 24
    if len(hourly) != expected:
        raise RuntimeError(f"Fortran hourly record count mismatch: expected {expected}, got {len(hourly)}")
    return hourly


def build_jax_params(run_kwargs: dict[str, Any]) -> tuple[SoilHydroParams, AeroParams, CanopySnowParams]:
    soil_params = SoilHydroParams(
        watsat=jnp.array(run_kwargs["watsat"]),
        watres=jnp.array(run_kwargs["watres"]),
        alpha_van=jnp.array(run_kwargs["alpha_van"]),
        n_van=jnp.array(run_kwargs["n_van"]),
        ksat=jnp.array(run_kwargs["ksat"]),
    )
    aero_params = AeroParams(
        hc=jnp.array(run_kwargs["hc"]),
        zmeas=jnp.array(run_kwargs["zmeas"]),
        zground=jnp.array(run_kwargs["zground"]),
        zo_ground=jnp.array(run_kwargs["zo_ground"]),
        w_leaf=jnp.array(run_kwargs["w_leaf"]),
    )
    cs_params = CanopySnowParams(
        wmax=jnp.array(run_kwargs["wmax"]),
        wmaxsnow=jnp.array(run_kwargs["wmaxsnow"]),
        kmelt=jnp.array(run_kwargs["kmelt"]),
        kfreeze=jnp.array(run_kwargs["kfreeze"]),
        frac_snowliq=jnp.array(run_kwargs["frac_snowliq"]),
        gsoil=jnp.array(run_kwargs["gsoil"]),
    )
    return soil_params, aero_params, cs_params


def make_jax_record(
    *,
    ref: dict[str, Any],
    day: int,
    hour_index: int,
    hour_of_day: int,
    temp_c: float,
    rg: float,
    prec: float,
    vpd: float,
    pres: float,
    co2: float,
    wind: float,
    lai: float,
    fapar: float,
    psi_soil_in: float,
    aj: float,
    gs: float,
    vcmax_hr: float,
    tr_phydro_mm: float,
    met_temp_smooth: float,
    met_prec_smooth: float,
    cw_in,
    cw_flux,
    cw_out,
    sw_in,
    sw_flux,
    sw_out,
) -> dict[str, Any]:
    return {
        "fn": "integration_hourly",
        "inputs": {
            "day": day,
            "hour_index": hour_index,
            "hour_of_day": hour_of_day,
            "date": ref["daily"]["dates"][day - 1],
            "timestamp": ref["hourly"]["timestamps"][hour_index - 1],
            "temp_c": temp_c,
            "rg": rg,
            "prec": prec,
            "vpd": vpd,
            "pres": pres,
            "co2": co2,
            "wind": wind,
            "lai": lai,
            "fapar": fapar,
        },
        "output": {
            "psi_soil_in": psi_soil_in,
            "aj": aj,
            "gs": gs,
            "vcmax_hr": vcmax_hr,
            "tr_phydro_mm": tr_phydro_mm,
            "met_temp_smooth": met_temp_smooth,
            "met_prec_smooth": met_prec_smooth,
            "cw_in_canopy_storage": scalar(cw_in.CanopyStorage),
            "cw_in_swe": scalar(cw_in.SWE),
            "cw_in_swe_i": scalar(cw_in.swe_i),
            "cw_in_swe_l": scalar(cw_in.swe_l),
            "cw_throughfall": scalar(cw_flux.Throughfall),
            "cw_interception": scalar(cw_flux.Interception),
            "cw_canopy_evap": scalar(cw_flux.CanopyEvap),
            "cw_soil_evap": scalar(cw_flux.SoilEvap),
            "cw_pot_infiltration": scalar(cw_flux.PotInfiltration),
            "cw_melt": scalar(cw_flux.Melt),
            "cw_freeze": scalar(cw_flux.Freeze),
            "cw_mbe": scalar(cw_flux.mbe),
            "cw_out_canopy_storage": scalar(cw_out.CanopyStorage),
            "cw_out_swe": scalar(cw_out.SWE),
            "cw_out_swe_i": scalar(cw_out.swe_i),
            "cw_out_swe_l": scalar(cw_out.swe_l),
            "sw_in_wat_sto": scalar(sw_in.WatSto),
            "sw_in_pond_sto": scalar(sw_in.PondSto),
            "sw_in_wliq": scalar(sw_in.Wliq),
            "sw_in_psi": scalar(sw_in.Psi),
            "sw_in_kh": scalar(sw_in.Kh),
            "sw_in_beta": scalar(sw_in.beta),
            "sw_infiltration": scalar(sw_flux.Infiltration),
            "sw_drainage": scalar(sw_flux.Drainage),
            "sw_runoff": scalar(sw_flux.Runoff),
            "sw_lateral_flow": scalar(sw_flux.LateralFlow),
            "sw_et": scalar(sw_flux.ET),
            "sw_mbe": scalar(sw_flux.mbe),
            "sw_out_wat_sto": scalar(sw_out.WatSto),
            "sw_out_pond_sto": scalar(sw_out.PondSto),
            "sw_out_wliq": scalar(sw_out.Wliq),
            "sw_out_psi": scalar(sw_out.Psi),
            "sw_out_kh": scalar(sw_out.Kh),
            "sw_out_beta": scalar(sw_out.beta),
        },
    }


def collect_jax_hourly(ref: dict[str, Any], start_day: int, end_day: int) -> list[dict[str, Any]]:
    # Build kwargs for the full range (up to end_day)
    run_kwargs = build_qvidja_run_kwargs(ref, end_day)
    soil_params, aero_params, cs_params = build_jax_params(run_kwargs)

    prefix_days = start_day - 1

    if prefix_days > 0:
        # ── JIT-compiled prefix via jax.lax.scan ──
        # Run the full integration for the prefix to get the carry state.
        # This is orders of magnitude faster than the eager Python loop.
        print(
            f"  JIT-compiling and running {prefix_days}-day prefix via jax.lax.scan...",
            flush=True,
        )
        prefix_kwargs = build_qvidja_run_kwargs(ref, prefix_days)
        prefix_carry, _ = jax.jit(
            run_integration, static_argnames=("phydro_optimizer", "invert_option")
        )(**prefix_kwargs)

        # Extract hydro state from prefix carry
        cw_state = prefix_carry.cw_state
        sw_state = prefix_carry.sw_state
        met_rolling = prefix_carry.met_rolling
        is_first_met = False
    else:
        cw_state, sw_state = initialization_spafhy(
            run_kwargs["soil_depth"],
            run_kwargs["max_poros"],
            run_kwargs["fc"],
            run_kwargs["maxpond"],
            soil_params,
        )
        met_rolling = jnp.zeros(2, dtype=jnp.float64)
        is_first_met = True

    # ── Eager loop for the window only ──
    window_days = end_day - start_day + 1
    print(
        f"  Eager-looping {window_days} window days ({window_days * 24} hours)...",
        flush=True,
    )
    records: list[dict[str, Any]] = []

    for day_offset in range(window_days):
        day_idx = start_day - 1 + day_offset  # 0-based index into run_kwargs arrays
        lai = run_kwargs["daily_lai"][day_idx]
        fapar = 1.0 - jnp.exp(-_K_EXT * lai)

        for hour_idx in range(24):
            hour_index = day_idx * 24 + hour_idx + 1
            temp_k = run_kwargs["hourly_temp"][day_idx, hour_idx]
            rg = run_kwargs["hourly_rg"][day_idx, hour_idx]
            prec = run_kwargs["hourly_prec"][day_idx, hour_idx]
            vpd = run_kwargs["hourly_vpd"][day_idx, hour_idx]
            pres = run_kwargs["hourly_pres"][day_idx, hour_idx]
            co2 = run_kwargs["hourly_co2"][day_idx, hour_idx]
            wind = run_kwargs["hourly_wind"][day_idx, hour_idx]
            tc = temp_k - 273.15

            psi_soil = sw_state.Psi
            ppfd = rg * 2.1 / jnp.maximum(lai, _LAI_GUARD)
            co2_ppm = co2 * 1.0e6
            phydro = pmodel_hydraulics_numerical(
                tc,
                ppfd,
                vpd,
                co2_ppm,
                pres,
                fapar,
                psi_soil,
                rdark_leaf=jnp.array(run_kwargs["rdark"]),
                conductivity=run_kwargs["conductivity"],
                psi50=run_kwargs["psi50"],
                b_param=run_kwargs["b_param"],
                alpha=run_kwargs["alpha_cost"],
                gamma_cost=run_kwargs["gamma_cost"],
                solver_kind="projected_lbfgs",
            )

            aj = phydro["aj"]
            gs = phydro["gs"]
            vcmax_hr = phydro["vcmax"]
            has_light = (lai > _LAI_GUARD) & (rg > 0.0)
            aj_guarded = jnp.where(has_light, aj, 0.0)
            gs_guarded = jnp.where(has_light, gs, 0.0)
            vcmax_guarded = jnp.where(has_light, vcmax_hr, 0.0)

            rho_w = density_h2o(tc, pres)
            tr_raw = 1.6 * gs_guarded * (vpd / pres) * 18.01528 / rho_w * lai
            tr_phydro = jnp.where(jnp.isfinite(gs_guarded) & has_light, tr_raw, 0.0)

            cw_in = cw_state
            sw_in = sw_state
            rn = rg * 0.7
            cw_state, cw_flux = canopy_water_flux(
                rn,
                tc,
                prec,
                vpd,
                wind,
                pres,
                fapar,
                lai,
                cw_state,
                sw_state.beta,
                sw_state.WatSto,
                aero_params,
                cs_params,
                jnp.array(1.0),
            )
            tr_spafhy = tr_phydro * SECONDS_PER_HOUR
            sw_state, sw_flux, _tr_out, _evap_out, _latflow_out = soil_water(
                sw_state,
                soil_params,
                jnp.array(run_kwargs["max_poros"]),
                cw_flux.PotInfiltration,
                tr_spafhy,
                cw_flux.SoilEvap,
                jnp.array(0.0),
                jnp.array(1.0),
            )

            met_daily = jnp.array([
                tc,
                prec + cw_flux.Melt / SECONDS_PER_HOUR,
            ])
            if is_first_met:
                met_rolling = met_daily
                is_first_met = False
            else:
                met_rolling = jnp.array([
                    _ALPHA_SMOOTH1 * met_daily[0] + (1.0 - _ALPHA_SMOOTH1) * met_rolling[0],
                    _ALPHA_SMOOTH2 * met_daily[1] + (1.0 - _ALPHA_SMOOTH2) * met_rolling[1],
                ])

            day_num = day_idx + 1
            records.append(
                make_jax_record(
                    ref=ref,
                    day=day_num,
                    hour_index=hour_index,
                    hour_of_day=hour_idx + 1,
                    temp_c=scalar(tc),
                    rg=scalar(rg),
                    prec=scalar(prec),
                    vpd=scalar(vpd),
                    pres=scalar(pres),
                    co2=scalar(co2),
                    wind=scalar(wind),
                    lai=scalar(lai),
                    fapar=scalar(fapar),
                    psi_soil_in=scalar(psi_soil),
                    aj=scalar(aj_guarded),
                    gs=scalar(gs_guarded),
                    vcmax_hr=scalar(vcmax_guarded),
                    tr_phydro_mm=scalar(tr_spafhy),
                    met_temp_smooth=scalar(met_rolling[0]),
                    met_prec_smooth=scalar(met_rolling[1]),
                    cw_in=cw_in,
                    cw_flux=cw_flux,
                    cw_out=cw_state,
                    sw_in=sw_in,
                    sw_flux=sw_flux,
                    sw_out=sw_state,
                )
            )

    expected = (end_day - start_day + 1) * 24
    if len(records) != expected:
        raise RuntimeError(f"JAX hourly record count mismatch: expected {expected}, got {len(records)}")
    return records


def write_metadata(path: Path, *, start_date: str, end_date: str, start_day: int, end_day: int,
                   fortran_count: int, jax_count: int, collection_stage: str) -> None:
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/collect_hourly_logs.py",
        "git_commit": comparison.get_git_commit(),
        "git_dirty": comparison.get_git_dirty(),
        "collection_stage": collection_stage,
        "qvidja_reference": str(QVIDJA_REF.relative_to(REPO_ROOT)),
        "start_date": start_date,
        "end_date": end_date,
        "start_day": start_day,
        "end_day": end_day,
        "fortran_records": fortran_count,
        "jax_records": jax_count,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ref = load_reference()
    start_day, end_day = resolve_day_window(ref, args.start_date, args.end_date)

    output_dir = args.output_dir or default_output_dir(args.start_date, args.end_date)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fortran_path = output_dir / "fortran-hourly.jsonl"
    jax_path = output_dir / "jax-hourly.jsonl"
    metadata_path = output_dir / "metadata.json"

    print(
        f"Collecting hourly logs for days {start_day}..{end_day} "
        f"({args.start_date}..{args.end_date})",
        flush=True,
    )
    print("Stage 1/2: collecting Fortran hourly logs via the extended harness...", flush=True)
    fortran_records = collect_fortran_hourly(ref, start_day, end_day)
    write_jsonl(fortran_path, fortran_records)
    write_metadata(
        metadata_path,
        start_date=args.start_date,
        end_date=args.end_date,
        start_day=start_day,
        end_day=end_day,
        fortran_count=len(fortran_records),
        jax_count=0,
        collection_stage="fortran-complete",
    )
    print(f"Wrote {fortran_path} ({len(fortran_records)} records)", flush=True)

    print(
        "Stage 2/2: collecting JAX hourly logs by replaying the full prefix needed "
        "to reach the requested window...",
        flush=True,
    )
    jax_records = collect_jax_hourly(ref, start_day, end_day)
    write_jsonl(jax_path, jax_records)
    write_metadata(
        metadata_path,
        start_date=args.start_date,
        end_date=args.end_date,
        start_day=start_day,
        end_day=end_day,
        fortran_count=len(fortran_records),
        jax_count=len(jax_records),
        collection_stage="complete",
    )

    print(f"Wrote {jax_path} ({len(jax_records)} records)", flush=True)
    print(f"Wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()