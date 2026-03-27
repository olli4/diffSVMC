#!/usr/bin/env python3
"""
Generate reference fixture files from the Fortran SVMC harness.

Builds the Fortran harness, runs it, reads the JSONL file it produces,
and writes structured JSON fixture files for phydro, water, and yasso tests.
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"


def build_harness() -> None:
    """Run make in the svmc-ref directory."""
    print("Building Fortran harness …")
    result = subprocess.run(
        ["make", "-j4"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("=== STDOUT ===", result.stdout, sep="\n")
        print("=== STDERR ===", result.stderr, sep="\n")
        sys.exit(1)
    print("Build OK")


def run_harness() -> list[dict]:
    """Execute the harness and parse its JSONL file output."""
    print("Running harness …")
    result = subprocess.run(
        [str(HERE / "harness")],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("=== STDERR ===", result.stderr, sep="\n")
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip())

    jsonl_path = HERE / "fixtures.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: harness did not produce {jsonl_path}")
        sys.exit(1)

    records: list[dict] = []
    for i, line in enumerate(jsonl_path.read_text().strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"Bad JSON on line {i}: {exc}\n  {line!r}")
            sys.exit(1)

    print(f"Parsed {len(records)} records from {jsonl_path.name}")
    return records


PHYDRO_FUNCTIONS = {
    "ftemp_arrh",
    "gammastar",
    "ftemp_kphio_c3",
    "ftemp_kphio_c4",
    "density_h2o",
    "viscosity_h2o",
    "calc_kmm",
    "scale_conductivity",
    "calc_gs",
    "calc_assim_light_limited",
    "fn_profit",
    "quadratic",
    "pmodel_hydraulics_numerical",
}

WATER_FUNCTIONS = {
    "e_sat",
    "penman_monteith",
    "soil_water_retention_curve",
    "soil_hydraulic_conductivity",
    "aerodynamics",
    "exponential_smooth_met",
    "canopy_water_snow",
    "ground_evaporation",
    "canopy_water_flux",
    "soil_water",
}

YASSO_FUNCTIONS = {
    "inputs_to_fractions",
    "matrixnorm",
    "matrixexp",
}


def split_and_write(records: list[dict]) -> None:
    """Split records into phydro/water and write JSON fixture files."""
    FIXTURES_DIR.mkdir(exist_ok=True)

    phydro: dict[str, list] = defaultdict(list)
    water: dict[str, list] = defaultdict(list)
    yasso: dict[str, list] = defaultdict(list)

    for rec in records:
        fn = rec["fn"]
        if fn in PHYDRO_FUNCTIONS:
            phydro[fn].append({"inputs": rec["inputs"], "output": rec["output"]})
        elif fn in WATER_FUNCTIONS:
            water[fn].append({"inputs": rec["inputs"], "output": rec["output"]})
        elif fn in YASSO_FUNCTIONS:
            yasso[fn].append({"inputs": rec["inputs"], "output": rec["output"]})
        else:
            print(f"WARNING: unknown function '{fn}', skipping")

    phydro_path = FIXTURES_DIR / "phydro.json"
    water_path = FIXTURES_DIR / "water.json"
    yasso_path = FIXTURES_DIR / "yasso.json"

    phydro_path.write_text(json.dumps(dict(phydro), indent=2) + "\n")
    water_path.write_text(json.dumps(dict(water), indent=2) + "\n")
    yasso_path.write_text(json.dumps(dict(yasso), indent=2) + "\n")

    print(f"Wrote {phydro_path}  ({sum(len(v) for v in phydro.values())} cases)")
    print(f"Wrote {water_path}  ({sum(len(v) for v in water.values())} cases)")
    print(f"Wrote {yasso_path}  ({sum(len(v) for v in yasso.values())} cases)")


def main() -> None:
    build_harness()
    records = run_harness()
    split_and_write(records)
    print("Done.")


if __name__ == "__main__":
    main()
