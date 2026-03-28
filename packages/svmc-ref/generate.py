#!/usr/bin/env python3
"""
Generate reference fixture files from the Fortran SVMC harness.

Builds the Fortran harness, runs it, reads the JSONL file it produces,
and writes structured JSON fixture files for phydro, water, yasso, and
allocation tests.

The staged ``src/`` and ``app/`` trees under ``packages/svmc-ref`` exist so
``fpm`` can build a small, reviewable mirror of the upstream Fortran sources.
Behavioral edits belong in ``vendor/SVMC/src`` and ``packages/svmc-ref/``'s
authoritative harness sources, not in the staged copies.
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"
STAGED_SOURCES_MANIFEST = HERE / "staged-sources.json"


def load_staged_sources_manifest() -> list[dict[str, str]]:
    """Load and validate the machine-readable staging manifest."""
    try:
        manifest = json.loads(STAGED_SOURCES_MANIFEST.read_text())
    except FileNotFoundError:
        print(f"ERROR: missing staging manifest {STAGED_SOURCES_MANIFEST}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: bad staging manifest JSON: {exc}")
        sys.exit(1)

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        print("ERROR: staging manifest must contain a non-empty 'entries' array")
        sys.exit(1)

    seen_targets: set[str] = set()
    validated_entries: list[dict[str, str]] = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            print(f"ERROR: manifest entry {index} is not an object")
            sys.exit(1)

        source = entry.get("source")
        target = entry.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            print(f"ERROR: manifest entry {index} must define string 'source' and 'target'")
            sys.exit(1)

        if target in seen_targets:
            print(f"ERROR: duplicate staged target '{target}' in manifest")
            sys.exit(1)
        seen_targets.add(target)

        target_path = Path(target)
        if target_path.parts[0] not in {"src", "app"}:
            print(f"ERROR: staged target '{target}' must live under src/ or app/")
            sys.exit(1)

        source_path = HERE.parent.parent / source
        if not source_path.exists():
            print(f"ERROR: staged source '{source}' does not exist")
            sys.exit(1)

        validated_entries.append(entry)

    return validated_entries


def build_harness() -> None:
    """Build harness using fpm."""
    import shutil

    print("Building Fortran harness …")
    staged_entries = load_staged_sources_manifest()
    # Stage the exact subset of upstream sources required by the harness.
    # The vendor tree remains the numerical source of truth.
    for entry in staged_entries:
        source_path = HERE.parent.parent / entry["source"]
        target_path = HERE / entry["target"]
        target_path.parent.mkdir(exist_ok=True)
        shutil.copy2(source_path, target_path)
    # Get netcdf include flags (may be empty if not installed)
    import subprocess
    nf_flags = ""
    try:
        nf_flags = subprocess.run(
            ["nf-config", "--fflags"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    fpm_flags = f"-O2 -g -m64 -freal-4-real-8 -fcheck=all {nf_flags}".strip()
    result = subprocess.run(
        ["fpm", "build", "--flag", fpm_flags],
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
    # Get netcdf include flags
    nf_flags = ""
    try:
        nf_flags = subprocess.run(
            ["nf-config", "--fflags"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    fpm_flags = f"-O2 -g -m64 -freal-4-real-8 -fcheck=all {nf_flags}".strip()
    result = subprocess.run(
        ["fpm", "run", "--flag", fpm_flags],
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
    "mod5c20",
    "decompose",
    "initialize_totc",
}

ALLOCATION_FUNCTIONS = {
    "alloc_hypothesis_2",
    "invert_alloc",
}


def split_and_write(records: list[dict]) -> None:
    """Split records into phydro/water and write JSON fixture files."""
    FIXTURES_DIR.mkdir(exist_ok=True)

    phydro: dict[str, list] = defaultdict(list)
    water: dict[str, list] = defaultdict(list)
    yasso: dict[str, list] = defaultdict(list)
    alloc: dict[str, list] = defaultdict(list)

    for rec in records:
        fn = rec["fn"]
        if fn in PHYDRO_FUNCTIONS:
            phydro[fn].append({"inputs": rec["inputs"], "output": rec["output"]})
        elif fn in WATER_FUNCTIONS:
            water[fn].append({"inputs": rec["inputs"], "output": rec["output"]})
        elif fn in YASSO_FUNCTIONS:
            yasso[fn].append({"inputs": rec["inputs"], "output": rec["output"]})
        elif fn in ALLOCATION_FUNCTIONS:
            alloc[fn].append({"inputs": rec["inputs"], "output": rec["output"]})
        else:
            print(f"WARNING: unknown function '{fn}', skipping")

    phydro_path = FIXTURES_DIR / "phydro.json"
    water_path = FIXTURES_DIR / "water.json"
    yasso_path = FIXTURES_DIR / "yasso.json"
    alloc_path = FIXTURES_DIR / "allocation.json"

    phydro_path.write_text(json.dumps(dict(phydro), indent=2) + "\n")
    water_path.write_text(json.dumps(dict(water), indent=2) + "\n")
    yasso_path.write_text(json.dumps(dict(yasso), indent=2) + "\n")
    alloc_path.write_text(json.dumps(dict(alloc), indent=2) + "\n")

    print(f"Wrote {phydro_path}  ({sum(len(v) for v in phydro.values())} cases)")
    print(f"Wrote {water_path}  ({sum(len(v) for v in water.values())} cases)")
    print(f"Wrote {yasso_path}  ({sum(len(v) for v in yasso.values())} cases)")
    print(f"Wrote {alloc_path}  ({sum(len(v) for v in alloc.values())} cases)")


def main() -> None:
    build_harness()
    records = run_harness()
    split_and_write(records)
    print("Done.")


if __name__ == "__main__":
    main()
