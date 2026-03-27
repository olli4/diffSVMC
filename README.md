# diffSVMC

Differentiable port of the [SVMC](https://github.com/huitang-earth/SVMC) vegetation process model to [JAX](https://github.com/jax-ml/jax) (Python) and [`@hamk-uas/jax-js-nonconsuming`](https://www.npmjs.com/package/@hamk-uas/jax-js-nonconsuming) (TypeScript/browser).

The goal is a numerically faithful, fully differentiable reimplementation suitable for gradient-based parameter calibration and interactive browser-based exploration.

🤖 AI generated code & documentation with gentle human supervision.

## Repository structure

```
packages/
  svmc-ref/    Fortran reference test harness (driver program that exercises
               original routines), fixture generation, branch audit
  svmc-jax/    JAX reimplementation (float64, autodiff, invariant tests)
  svmc-js/     TypeScript reimplementation (float32/float64, browser-ready)
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
- Phase 4 Yasso validation is still in progress. The current `matrixExp` coverage now mixes analytic sanity cases and matrices generated through the real `yasso20.mod5c20` coefficient-matrix path, but full wrapper-boundary yearly Yasso fixtures are still pending.

## Current status

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Project foundation | ✅ Complete |
| 1 | All leaf functions | ✅ Complete |
| 2 | P-Hydro assemblies & optimizer | ✅ Complete |
| 3 | SpaFHy canopy & soil water balance | ✅ Complete |
| 4 | Carbon allocation & Yasso20 | Planned |
| 5 | Main SVMC integration loop | Planned |
| 6 | Interactive web application | Planned |

## Quick start

### Prerequisites

- Node.js ≥ 20, pnpm
- Python ≥ 3.11, pip
- gfortran (for fixture regeneration only)

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

The `vendor/SVMC/` directory contains the original SVMC Fortran model as a git submodule, copyright (c) 2023 huitang-earth, licensed under the MIT License. See [`vendor/SVMC/LICENSE`](vendor/SVMC/LICENSE).

## License

[MIT](LICENSE)
