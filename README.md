# diffSVMC

Differentiable port of the [SVMC](https://github.com/huitang-earth/SVMC) vegetation process model to [JAX](https://github.com/jax-ml/jax) (Python) and [`@hamk-uas/jax-js-nonconsuming`](https://www.npmjs.com/package/@hamk-uas/jax-js-nonconsuming) (TypeScript/browser).

The goal is a numerically faithful, fully differentiable reimplementation suitable for gradient-based parameter calibration and interactive browser-based exploration.

🤖 AI generated code & documentation with gentle human supervision.

## Repository structure

```
packages/
  svmc-ref/    Fortran reference harness, fixture generation, branch audit
  svmc-jax/    JAX reimplementation (float64, autodiff, invariant tests)
  svmc-js/     TypeScript reimplementation (float32/float64, browser-ready)
vendor/
  SVMC/        Original Fortran model (git submodule)
scripts/       CI tooling (branch coverage audit)
issues/        Upstream bug reports for SVMC
```

## Porting approach

Each submodel is ported bottom-up (leaf functions first), following the phased plan in [PLAN.md](PLAN.md):

1. **Fortran reference logging** — harness captures inputs/outputs as JSONL fixtures.
2. **Branch audit** — every conditional branch is annotated with `PORT-BRANCH` tags, registered, and evaluated for fixture coverage.
3. **JAX port** — differentiable reimplementation with `jnp.where` for branch-free autodiff. Tested with fixture playback, metamorphic invariants, and gradient/OOD validation.
4. **TypeScript port** — `@hamk-uas/jax-js-nonconsuming` reimplementation tested against the same fixtures, with `checkLeaks` memory safety and epsilon-derived tolerance bounds.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the `PORT-BRANCH` convention.

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
