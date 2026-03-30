# diffSVMC

Differentiable port of the [SVMC](https://github.com/huitang-earth/SVMC) vegetation process model to [JAX](https://github.com/jax-ml/jax) (Python) and [`@hamk-uas/jax-js-nonconsuming`](https://hamk-uas.github.io/jax-js-nonconsuming) (TypeScript/browser).

The goal is a numerically faithful, fully differentiable reimplementation suitable for gradient-based parameter calibration and interactive browser-based exploration.

🤖 AI generated code & documentation with gentle human supervision.

## Repository structure

```
packages/
  svmc-ref/    Fortran reference harness, fixture generation, branch audit,
               and the fpm-staged source mirror used for reference builds
  svmc-jax/    JAX reimplementation (float64, autodiff, invariant tests)
  svmc-js/     TypeScript reimplementation (float32/float64, browser-ready)
  svmc-webr/   R/WebR wrapper for running Fortran allocation in the browser
website/       Interactive demo (Vite, served on GitHub Pages)
vendor/
  SVMC/        Original Fortran model (git submodule)
scripts/       CI tooling (branch coverage audit)
issues/        Upstream bug reports for SVMC
```

## Porting approach

Each submodel is ported bottom-up (leaf functions — the lowest-level routines with no internal sub-calls — first), following the phased plan in [PLAN.md](PLAN.md):

1. **Fortran reference logging** — the test harness (a driver program that calls each original Fortran routine over a grid of inputs) captures inputs and outputs as JSONL (newline-delimited JSON) fixtures (saved reference data used as ground-truth test cases).
2. **Branch audit** — every conditional branch in the Fortran source is annotated with a `PORT-BRANCH` tag, registered in a coverage manifest, and evaluated for whether the fixtures exercise it.
3. **JAX port** — differentiable reimplementation using `jnp.where` for branch-free autodiff (automatic differentiation). Tested with fixture playback, metamorphic invariants (tests that check mathematical properties like monotonicity or conservation rather than exact values), and gradient/out-of-distribution validation.
4. **TypeScript port** — `@hamk-uas/jax-js-nonconsuming` reimplementation tested against the same fixtures, with `checkLeaks` (runtime verification that every GPU array is properly disposed) memory safety and epsilon-derived (computed from the floating-point precision limit) tolerance bounds.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the `PORT-BRANCH` convention.

## Current deviations

- The TypeScript `matrixExp` helper currently uses a bounded masked-squaring policy (`MAX_J = 20`) because the present `lax.foriLoop` API requires a static loop bound. The current reference fixtures explicitly stay within that bound; widening or removing it is required before using the helper on materially larger matrix norms.
- Both the JAX and TypeScript `mod5c20` transient solvers split the `exp(At)·z₁ − b` computation into `exp(At)·(A·init) + (exp(At) − I)·b` and use a first-order Taylor approximation (`At·b`) for the second term when `‖At‖ ≤ √ε`. The Fortran reference computes `exp(At)·(A·init + b) − b` directly. The split avoids catastrophic cancellation when `‖At‖` is small (e.g. extreme cold), where the direct subtraction `exp(At)·b − b` loses the `O(At·b)` contribution. Boundary tests verify smooth transition across the threshold in both languages.

## Current status

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Project foundation | ✅ Complete |
| 1 | All leaf functions | ✅ Complete |
| 2 | P-Hydro assemblies & optimizer | ✅ Complete |
| 3 | SpaFHy canopy & soil water balance | ✅ Complete |
| 4 | Carbon allocation & Yasso20 | In progress |
| 5 | Main SVMC integration loop | Planned |
| 6 | Interactive web application | In progress (demo live) |

## Quick start

### Prerequisites

- Node.js ≥ 20, pnpm
- Python ≥ 3.11, pip
- gfortran
- fpm (for fixture regeneration)
- NetCDF C and Fortran development libraries, with `nf-config` on `PATH` when
  compiler include paths are not already configured

For the WebR build and native R testing:

- **R ≥ 4.3** with development headers
- **Docker** (for building the WASM R package locally)

On Ubuntu/Debian:

```bash
sudo apt-get install r-base r-base-dev
```

### Install and test

```bash
# Python (JAX tests + branch audit)
pip install -e .[dev]
pytest

# TypeScript (browser-based tests via Playwright)
pnpm install
pnpm vitest run

# Branch coverage audit
python scripts/verify_branch_coverage.py
```

### Regenerate reference fixtures

```bash
python packages/svmc-ref/generate.py
```

`packages/svmc-ref/generate.py` stages the exact maintained reference Fortran sources needed by
the harness into `packages/svmc-ref/src/` and `packages/svmc-ref/app/` before
building with `fpm`. The authoritative source-to-stage mapping lives in
`packages/svmc-ref/staged-sources.json`. Those staged copies are reviewable build
inputs, not a second source of truth. Make behavioral edits in `vendor/SVMC/src/`
or `packages/svmc-ref/harness.f90`, then rerun the generator.

Within this repository, `vendor/SVMC/` is the maintained Fortran reference base for
porting and fixture generation. It was seeded from the external
`huitang-earth/SVMC` project and may carry repo-local, non-numerical modifications
that support conservative porting work.

### Website (interactive demo)

The `website/` package provides an interactive browser demo of the full
SVMC model running via [WebR](https://docs.r-wasm.org/webr/latest/)
(Fortran compiled to WebAssembly). It replays the vendored Qvidja
reference inputs (ERA5-Land forcing, Sentinel-2 LAI, management events)
for 1697 days and renders live charts of GPP, NEE, carbon pools, soil
moisture, and stomatal conductance.

```bash
# Development server (port 5173, with COOP/COEP headers for SharedArrayBuffer)
pnpm -C website dev

# Production build (WebR WASM package + Vite)
pnpm build
```

On GitHub Pages, a `coi-serviceworker` provides the COOP/COEP headers
that SharedArrayBuffer requires (GitHub Pages cannot set custom headers).

### Performance

Full 1697-day Qvidja reference run (40,728 hourly steps) on an Intel
N100 mini-PC:

| Runtime | Time | Slowdown |
|---|---|---|
| Native R/Fortran | 1.6 s | 1× |
| WebR/WASM (Chromium) | 5.4 s | ~3.4× |

The first `pnpm build` uses Docker to compile the R package to WASM via
the `ghcr.io/r-wasm/webr:main` image. Subsequent builds skip this step
if `website/public/bin/` already exists. To force a rebuild:

```bash
rm -rf website/public/bin website/public/src
pnpm build
```

If `website/public/{bin,src}` are root-owned from a previous Docker build,
run the one-time ownership fix first:

```bash
sudo website/install-webr.sh
```

The build script (`website/build-webr.sh`) now runs the Docker container as
your current uid:gid, stages output through `tmp/webr-staging/`, and then
copies to `website/public/`. Normal builds should therefore stay user-owned.
The remaining reasons you might still need `sudo` are:

- your host Docker setup requires `sudo docker`
- `website/public/{bin,src}` or `tmp/webr-staging/` are already root-owned
  from an older build and need a one-time ownership fix

For local development without Docker, you can seed the CRAN repo from a
CI artifact:

```bash
# Download the artifact from GitHub Pages or CI, then:
tar xf artifact.tar -C website/public/
pnpm -C website dev
```

#### Testing the R package natively

You can also build and test the R package with native R (not WebR):

```bash
R CMD build packages/svmc-webr
R CMD INSTALL --library=tmp/R-lib SVMCwebr_0.1.0.tar.gz
R_LIBS=tmp/R-lib R -e 'library(SVMCwebr); str(alloc_hypothesis_2(
  temp_day=15, gpp_day=3e-7, leaf_rdark_day=3e-8, pft_type_code=1L))'
```

### Precision modes

The TypeScript port supports configurable numeric precision:

```bash
# Default: float32 (fast, GPU-friendly)
pnpm vitest run

# Higher accuracy: float64
SVMC_JS_DTYPE=float64 pnpm vitest run
```

## Authors

Olli Niemitalo (Olli.Niemitalo@hamk.fi) — Supervision of AI coding agents.

## Third-party code

The `vendor/SVMC/` directory contains this repository's maintained SVMC Fortran
reference tree as a git submodule. It derives from the external
`huitang-earth/SVMC` project and remains MIT-licensed; see `vendor/SVMC/LICENSE`.

The reference harness also stages selected Fortran sources from that maintained
reference tree into
`packages/svmc-ref/src/` for `fpm` builds. License provenance for those staged
copies, including the vendored L-BFGS-B BSD-3-Clause text, is documented in
`packages/svmc-ref/THIRD_PARTY_NOTICES.md`.

## License

[MIT](LICENSE)
