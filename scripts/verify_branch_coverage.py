#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "packages" / "svmc-ref" / "branch-coverage.json"
HARNESS_PATH = ROOT / "packages" / "svmc-ref" / "harness.f90"
VENDOR_DIR = ROOT / "vendor" / "SVMC" / "src"
FIXTURE_DIR = ROOT / "packages" / "svmc-ref" / "fixtures"
PORT_BRANCH_RE = re.compile(r"!\s*PORT-BRANCH:\s*([A-Za-z0-9_.-]+)")
WAIVER_KIND_REQUIREMENTS = {
    "fixture-gap": ("next_action",),
    "dead-code": ("evidence",),
    "fatal-path": ("safe_test_strategy",),
    "undefined-behavior": ("stabilization_plan",),
}


def fail(message: str) -> None:
    print(f"BRANCH AUDIT FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scan_branch_tags(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tags: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = PORT_BRANCH_RE.search(line)
        if not match:
            continue
        branch_id = match.group(1)
        tags.append((index, branch_id))

        condition_line = lines[index] if index < len(lines) else ""
        if not condition_line.lstrip().startswith("! Condition:"):
            fail(f"{path.relative_to(ROOT)}:{index} tag {branch_id!r} is missing the required '! Condition:' line immediately after it")
    return tags


def epsilon(value: float) -> float:
    if value == 0.0:
        return sys.float_info.epsilon
    return sys.float_info.epsilon * abs(value)


def penman_raw(case: dict[str, object]) -> float:
    inputs = case["inputs"]
    tc = float(inputs["tc"])
    patm = float(inputs["patm"])
    ae = float(inputs["AE"])
    vpd = float(inputs["vpd"])
    gs = float(inputs["Gs"])
    ga = float(inputs["Ga"])

    cp = 1004.67
    rho = 1.25
    lambda_ = 1.0e3 * (3147.5 - 2.37 * (tc + 273.15))
    esat = 1.0e3 * (0.6112 * pow(2.718281828459045, (17.67 * tc) / (tc + 273.16 - 29.66)))
    slope = 17.502 * 240.97 * esat / pow(240.97 + tc, 2)
    gamma = patm * cp / (0.622 * lambda_)
    return (slope * ae + rho * cp * ga * vpd) / (slope + gamma * (1.0 + ga / gs))


def compute_fixture_coverage() -> dict[str, bool]:
    phydro = load_json(FIXTURE_DIR / "phydro.json")
    water = load_json(FIXTURE_DIR / "water.json")
    yasso = load_json(FIXTURE_DIR / "yasso.json")

    assert isinstance(phydro, dict)
    assert isinstance(water, dict)
    assert isinstance(yasso, dict)

    c3_cases = list(phydro["ftemp_kphio_c3"])
    c4_cases = list(phydro["ftemp_kphio_c4"])
    kphio_cases = c3_cases + c4_cases
    quadratic_cases = list(phydro["quadratic"])
    fn_profit_cases = list(phydro["fn_profit"])
    soil_retention_cases = list(water["soil_water_retention_curve"])
    soil_conductivity_cases = list(water["soil_hydraulic_conductivity"])
    aerodynamics_cases = list(water["aerodynamics"])
    penman_cases = list(water["penman_monteith"])
    smooth_cases = list(water["exponential_smooth_met"])
    decompose_cases = list(yasso.get("decompose", []))
    has_pm_hypothesis = any(str(case["inputs"].get("hypothesis", "PM")) == "PM" for case in fn_profit_cases)
    has_non_pm_hypothesis = any(str(case["inputs"].get("hypothesis", "PM")) != "PM" for case in fn_profit_cases)
    has_do_optim_true = any(bool(case["inputs"].get("do_optim", False)) for case in fn_profit_cases)
    has_do_optim_false = any(not bool(case["inputs"].get("do_optim", False)) for case in fn_profit_cases)

    has_negative_discriminant = False
    has_nonnegative_discriminant = False
    has_near_zero_true = False
    has_near_zero_false = False
    has_impossible_true = False
    has_impossible_false = False
    has_linear_true = False
    has_linear_false = False
    has_zero_ab_true = False
    has_zero_ab_false = False
    for case in quadratic_cases:
        inputs = case["inputs"]
        a = float(inputs["a"])
        b = float(inputs["b"])
        c = float(inputs["c"])
        disc = b * b - 4.0 * a * c

        if disc < 0.0:
            has_negative_discriminant = True
            near_zero = -disc < 3.0 * epsilon(b)
            if near_zero:
                has_near_zero_true = True
                has_impossible_false = True
            else:
                has_near_zero_false = True
                has_impossible_true = True
        else:
            has_nonnegative_discriminant = True

        if a == 0.0:
            has_linear_true = True
            if b == 0.0:
                has_zero_ab_true = True
            else:
                has_zero_ab_false = True
        else:
            has_linear_false = True

    has_ground_cap_true = False
    has_ground_cap_false = False
    has_zn_cap_true = False
    has_zn_cap_false = False
    has_rb_guard_true = False
    has_rb_guard_false = False
    for case in aerodynamics_cases:
        inputs = case["inputs"]
        hc = float(inputs["hc"])
        zground = float(inputs["zground"])
        lai = float(inputs["LAI"])
        zg1 = min(zground, 0.1 * hc)

        if zground > 0.1 * hc:
            has_ground_cap_true = True
        else:
            has_ground_cap_false = True

        if zg1 / hc > 1.0:
            has_zn_cap_true = True
        else:
            has_zn_cap_false = True

        if lai > 1e-16:
            has_rb_guard_true = True
        else:
            has_rb_guard_false = True

    has_penman_floor_true = any(penman_raw(case) < 0.0 for case in penman_cases)
    has_penman_floor_false = any(penman_raw(case) >= 0.0 for case in penman_cases)

    has_invalid_ind_true = any(int(case["inputs"]["met_ind_in"]) < 1 for case in smooth_cases)
    has_invalid_ind_false = any(int(case["inputs"]["met_ind_in"]) >= 1 for case in smooth_cases)
    has_init_true = any(int(case["inputs"]["met_ind_in"]) == 1 for case in smooth_cases)
    has_init_false = any(int(case["inputs"]["met_ind_in"]) != 1 for case in smooth_cases)

    # --- Phase 3: ground_evaporation ---
    ground_evap_cases = list(water["ground_evaporation"])
    eps = 1e-16
    has_snow_floor_true = any(float(c["inputs"]["SWE"]) > eps for c in ground_evap_cases)
    has_snow_floor_false = any(float(c["inputs"]["SWE"]) <= eps for c in ground_evap_cases)

    # --- Phase 3: canopy_water_snow ---
    cws_cases = list(water["canopy_water_snow"])
    tmin = 0.0
    tmax = 1.0
    tmelt = 0.0

    has_precip_snow = any(float(c["inputs"]["T"]) <= tmin for c in cws_cases)
    has_precip_rain = any(float(c["inputs"]["T"]) >= tmax for c in cws_cases)

    has_lai_guard_true = any(float(c["inputs"]["LAI"]) > eps for c in cws_cases)
    has_lai_guard_false = any(float(c["inputs"]["LAI"]) <= eps for c in cws_cases)

    has_precip_mixed = any(tmin < float(c["inputs"]["T"]) < tmax for c in cws_cases)

    has_sublim = any(
        float(c["inputs"]["Pre"]) == 0.0 and float(c["inputs"]["T"]) <= tmin and float(c["inputs"]["LAI"]) > eps
        for c in cws_cases
    )
    has_evap = any(
        float(c["inputs"]["Pre"]) == 0.0 and float(c["inputs"]["T"]) > tmin and float(c["inputs"]["LAI"]) > eps
        for c in cws_cases
    )
    has_no_evap = any(
        float(c["inputs"]["Pre"]) > 0.0 and float(c["inputs"]["LAI"]) > eps
        for c in cws_cases
    )

    has_unload_true = any(float(c["inputs"]["T"]) >= tmin for c in cws_cases)
    has_unload_false = any(float(c["inputs"]["T"]) < tmin for c in cws_cases)

    has_interc_snow = any(float(c["inputs"]["T"]) < tmin for c in cws_cases)
    has_interc_liquid = any(float(c["inputs"]["T"]) >= tmin for c in cws_cases)

    has_melt = any(float(c["inputs"]["T"]) >= tmelt for c in cws_cases)
    has_freeze = any(
        float(c["inputs"]["T"]) < tmelt and float(c["inputs"]["swe_l_in"]) > 0.0
        for c in cws_cases
    )
    has_no_phase_change = any(
        float(c["inputs"]["T"]) < tmelt and float(c["inputs"]["swe_l_in"]) <= 0.0
        for c in cws_cases
    )

    has_totc_threshold_true = False
    has_totc_threshold_false = False
    has_nc_h_unusual_true = False
    has_nc_h_unusual_false = False
    has_cue_upper_true = False
    has_cue_upper_false = False
    has_cue_lower_true = False
    has_cue_lower_false = False
    for case in decompose_cases:
        inputs = case["inputs"]
        cstate = [float(value) for value in inputs["cstate"]]
        nstate = float(inputs["nstate"])
        totc = sum(cstate)
        h_pool = cstate[4]

        if totc < 1e-6:
            has_totc_threshold_true = True
            continue
        has_totc_threshold_false = True

        if h_pool * 0.1 > nstate:
            has_nc_h_unusual_true = True
        else:
            has_nc_h_unusual_false = True

        nc_som = nstate / totc
        raw_cue = 0.43 * (nc_som / 0.1) ** 0.6
        capped_cue = min(raw_cue, 1.0)

        if raw_cue > 1.0:
            has_cue_upper_true = True
        else:
            has_cue_upper_false = True

        if capped_cue < 0.1:
            has_cue_lower_true = True
        else:
            has_cue_lower_false = True

    coverage = {
        "phydro.ftemp_kphio.c4_select": bool(c3_cases) and bool(c4_cases),
        "phydro.ftemp_kphio.negative_clamp": any(float(case["output"]) == 0.0 for case in kphio_cases)
        and any(float(case["output"]) > 0.0 for case in kphio_cases),
        "phydro.quadratic.negative_discriminant": has_negative_discriminant and has_nonnegative_discriminant,
        "phydro.quadratic.near_zero_discriminant": has_near_zero_true and has_near_zero_false,
        "phydro.quadratic.impossible_discriminant": has_impossible_true and has_impossible_false,
        "phydro.quadratic.linear_fallback": has_linear_true and has_linear_false,
        "phydro.quadratic.zero_ab": has_zero_ab_true and has_zero_ab_false,
        "phydro.fn_profit.hypothesis_select": has_pm_hypothesis and has_non_pm_hypothesis,
        "phydro.fn_profit.optim_negate": has_do_optim_true and has_do_optim_false,
        "water.soil_retention.porosity_floor": any(
            float(case["inputs"]["watsat"]) < 0.01 for case in soil_retention_cases
        ) and any(float(case["inputs"]["watsat"]) >= 0.01 for case in soil_retention_cases),
        "water.soil_conductivity.porosity_floor": any(
            float(case["inputs"]["watsat"]) < 0.01 for case in soil_conductivity_cases
        ) and any(float(case["inputs"]["watsat"]) >= 0.01 for case in soil_conductivity_cases),
        "water.aerodynamics.ground_height_cap": has_ground_cap_true and has_ground_cap_false,
        "water.aerodynamics.zn_cap": has_zn_cap_true and has_zn_cap_false,
        "water.aerodynamics.rb_lai_guard": has_rb_guard_true and has_rb_guard_false,
        "water.penman_monteith.le_floor": has_penman_floor_true and has_penman_floor_false,
        "yasso.exponential_smooth_met.invalid_ind_guard": has_invalid_ind_true and has_invalid_ind_false,
        "yasso.exponential_smooth_met.init_vs_smooth": has_init_true and has_init_false,
        "water.ground_evaporation.snow_floor_zero": has_snow_floor_true and has_snow_floor_false,
        "water.canopy_water_snow.precip_phase": has_precip_snow and has_precip_rain and has_precip_mixed,
        "water.canopy_water_snow.lai_evap_guard": has_lai_guard_true and has_lai_guard_false,
        "water.canopy_water_snow.sublim_vs_evap": has_sublim and has_evap and has_no_evap,
        "water.canopy_water_snow.snow_unloading": has_unload_true and has_unload_false,
        "water.canopy_water_snow.interception_phase": has_interc_snow and has_interc_liquid,
        "water.canopy_water_snow.melt_freeze": has_melt and has_freeze and has_no_phase_change,
        "yasso.decompose.totc_threshold": has_totc_threshold_true and has_totc_threshold_false,
        "yasso.decompose.nc_h_unusual": has_nc_h_unusual_true and has_nc_h_unusual_false,
        "yasso.decompose.cue_upper_cap": has_cue_upper_true and has_cue_upper_false,
        "yasso.decompose.cue_lower_floor": has_cue_lower_true and has_cue_lower_false,
    }
    return coverage


def require_fields(entry: dict[str, object], fields: Iterable[str], branch_id: str) -> None:
    missing = [field for field in fields if field not in entry]
    if missing:
        fail(f"registry entry {branch_id!r} is missing required fields: {', '.join(missing)}")


def validate_waiver(branch_id: str, waiver: dict[str, object]) -> str:
    for waiver_field in ("scope", "approved_by", "reason", "kind"):
        waiver_value = waiver.get(waiver_field)
        if not isinstance(waiver_value, str) or not waiver_value.strip():
            fail(f"registry entry {branch_id!r} has an invalid waiver.{waiver_field}")

    kind = str(waiver["kind"])
    if kind not in WAIVER_KIND_REQUIREMENTS:
        allowed = ", ".join(sorted(WAIVER_KIND_REQUIREMENTS))
        fail(f"registry entry {branch_id!r} uses unknown waiver.kind {kind!r}; expected one of: {allowed}")

    for required_field in WAIVER_KIND_REQUIREMENTS[kind]:
        value = waiver.get(required_field)
        if not isinstance(value, str) or not value.strip():
            fail(f"registry entry {branch_id!r} with waiver.kind={kind!r} must include waiver.{required_field}")

    return kind


def main() -> None:
    if not VENDOR_DIR.exists():
        fail(f"vendor source directory is missing: {VENDOR_DIR.relative_to(ROOT)} (did you initialize submodules?)")

    registry = load_json(REGISTRY_PATH)
    if not isinstance(registry, dict):
        fail("branch-coverage.json root must be an object")

    branches = registry.get("branches")
    summary = registry.get("summary")
    if not isinstance(branches, list):
        fail("branch-coverage.json must contain a 'branches' array")
    if not isinstance(summary, dict):
        fail("branch-coverage.json must contain a 'summary' object")

    vendor_tag_locations: dict[str, list[str]] = {}
    for path in sorted(VENDOR_DIR.glob("*.f90")):
        for line_number, branch_id in scan_branch_tags(path):
            vendor_tag_locations.setdefault(branch_id, []).append(
                f"{path.relative_to(ROOT)}:{line_number}"
            )

    if not vendor_tag_locations:
        fail("no vendor PORT-BRANCH tags found")

    harness_tag_ids = {branch_id for _, branch_id in scan_branch_tags(HARNESS_PATH)}
    vendor_ids = set(vendor_tag_locations)
    if not harness_tag_ids.issubset(vendor_ids):
        extra = ", ".join(sorted(harness_tag_ids - vendor_ids))
        fail(f"harness duplicates reference non-vendor ids: {extra}")

    computed_coverage = compute_fixture_coverage()
    if set(computed_coverage) != vendor_ids:
        missing_eval = sorted(vendor_ids - set(computed_coverage))
        extra_eval = sorted(set(computed_coverage) - vendor_ids)
        problems: list[str] = []
        if missing_eval:
            problems.append(f"missing evaluators for {', '.join(missing_eval)}")
        if extra_eval:
            problems.append(f"extra evaluators for {', '.join(extra_eval)}")
        fail("fixture evaluator map is out of sync: " + "; ".join(problems))

    registry_by_id: dict[str, dict[str, object]] = {}
    for entry in branches:
        if not isinstance(entry, dict):
            fail("every branch registry entry must be an object")
        branch_id = entry.get("id")
        if not isinstance(branch_id, str):
            fail("every branch registry entry must have a string 'id'")
        if branch_id in registry_by_id:
            fail(f"duplicate registry id {branch_id!r}")
        registry_by_id[branch_id] = entry

    registry_ids = set(registry_by_id)
    if registry_ids != vendor_ids:
        missing = sorted(vendor_ids - registry_ids)
        extra = sorted(registry_ids - vendor_ids)
        problems: list[str] = []
        if missing:
            problems.append(f"missing registry entries for {', '.join(missing)}")
        if extra:
            problems.append(f"stale registry entries for {', '.join(extra)}")
        fail("vendor tags and registry ids differ: " + "; ".join(problems))

    required_fields = (
        "id",
        "file",
        "condition",
        "else_behavior",
        "fixture_both_sides",
        "jax_tested",
        "ts_tested",
        "notes",
    )
    uncovered_ids: list[str] = []
    waiver_kind_counts = {kind: 0 for kind in WAIVER_KIND_REQUIREMENTS}
    for branch_id, entry in registry_by_id.items():
        require_fields(entry, required_fields, branch_id)

        file_path = entry["file"]
        if not isinstance(file_path, str):
            fail(f"registry entry {branch_id!r} has non-string file path")
        if file_path != str(Path(file_path).as_posix()):
            fail(f"registry entry {branch_id!r} must use forward-slash relative paths")
        if branch_id not in vendor_tag_locations:
            fail(f"registry entry {branch_id!r} does not exist in vendor Fortran")
        if not any(location.startswith(file_path + ":") for location in vendor_tag_locations[branch_id]):
            locations = ", ".join(vendor_tag_locations[branch_id])
            fail(f"registry entry {branch_id!r} points to {file_path}, but the vendor tag is actually in {locations}")

        recorded = entry["fixture_both_sides"]
        if not isinstance(recorded, bool):
            fail(f"registry entry {branch_id!r} must use a boolean fixture_both_sides value")

        actual = computed_coverage[branch_id]
        if recorded != actual:
            fail(
                f"registry entry {branch_id!r} records fixture_both_sides={recorded}, but the computed value is {actual}"
            )

        waiver = entry.get("waiver")
        if actual:
            if waiver is not None:
                fail(f"registry entry {branch_id!r} is fully covered and must not carry a waiver")
        else:
            uncovered_ids.append(branch_id)
            if not isinstance(waiver, dict):
                fail(f"registry entry {branch_id!r} is uncovered and must include an explicit waiver object")
            waiver_kind = validate_waiver(branch_id, waiver)
            waiver_kind_counts[waiver_kind] += 1

    expected_summary = {
        "total_branches": len(registry_ids),
        "both_sides_covered": len(registry_ids) - len(uncovered_ids),
        "single_side_only": len(uncovered_ids),
        "waived_uncovered": len(uncovered_ids),
        "coverage_pct": f"{round((len(registry_ids) - len(uncovered_ids)) * 100 / len(registry_ids))}% ({len(registry_ids) - len(uncovered_ids)}/{len(registry_ids)})",
    }
    if summary != expected_summary:
        fail(f"summary mismatch: expected {expected_summary}, found {summary}")

    print(
        "Branch audit passed: "
        f"{len(registry_ids)} vendor tags registered, "
        f"{expected_summary['both_sides_covered']}/{expected_summary['total_branches']} branches fully covered, "
        f"{expected_summary['waived_uncovered']} uncovered branches explicitly waived."
    )
    if uncovered_ids:
        kinds = ", ".join(
            f"{kind}={count}" for kind, count in sorted(waiver_kind_counts.items()) if count
        )
        print(f"Waiver categories: {kinds}")


if __name__ == "__main__":
    main()