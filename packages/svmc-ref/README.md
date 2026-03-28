# svmc-ref

`svmc-ref` is the Fortran reference harness package used to regenerate fixture data
for the JAX and TypeScript ports.

## Source of truth

- `vendor/SVMC/src/` remains the authoritative upstream source tree.
- `packages/svmc-ref/harness.f90` remains the authoritative harness entry point.
- `packages/svmc-ref/src/` and `packages/svmc-ref/app/` are staged mirrors refreshed
  by `packages/svmc-ref/generate.py` so `fpm` can build the harness with a small,
  explicit input set.
- `packages/svmc-ref/staged-sources.json` is the machine-readable manifest that
  defines exactly which authoritative files are staged into those mirrors.
- Do not make behavioral edits in the staged copies. Edit the authoritative source,
  then rerun `python packages/svmc-ref/generate.py` to restage and regenerate fixtures.

## Build flow

1. `generate.py` reads `staged-sources.json`, validates the declared source and
  target paths, then stages the listed files into `src/` and `app/`.
2. `fpm` builds and runs the staged harness.
3. The harness writes `fixtures.jsonl`.
4. `generate.py` splits the JSONL stream into `fixtures/*.json` for downstream tests.

## Requirements

- `gfortran`
- `fpm`
- NetCDF C and Fortran development libraries (`netcdf`, `netcdff`)
- `nf-config` on `PATH` when compiler include paths are not already configured

## Third-party provenance

Selected staged files keep upstream license headers intact and are documented in
`packages/svmc-ref/THIRD_PARTY_NOTICES.md`.

- The staged L-BFGS-B support files use the BSD-3-Clause license text recorded in
  `packages/svmc-ref/LICENSES/BSD-3-Clause-L-BFGS-B.txt`.
- The staged `yassofortran20.f90` file retains its upstream GPL notice.