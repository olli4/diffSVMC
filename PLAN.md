# SVMC Differentiable Porting Plan

This document outlines the phased, bottom-up plan for porting the SVMC process model to JAX and TypeScript (via `@hamk-uas/jax-js-nonconsuming`).

## Definition of Done (Per Submodel)

To prevent technical debt and ensure a rigorous port, every function/submodel must meet these criteria before moving to the next:

1. **Fortran Reference Covered**: Logging captures baseline examples _and_ explicitly forces both single and combined branch-triggering conditions. For wrapper submodels, cross-level wrapper traces must be logged to validate composition (e.g., argument order, units). Any `PORT-BRANCH` added in vendor/harness/ports must be registered in `branch-coverage.json` and evaluated by `verify_branch_coverage.py` before the phase can close.
  Coverage claims must be branch-arm-specific: multi-arm conditionals are not considered covered unless fixtures exercise each materially distinct arm and the evaluator distinguishes them.
2. **Fixture & State Contract Established**: Reference JSON fixtures form a strict state-shape and interface contract across packages. Schema changes require simultaneous regeneration and alignment across `svmc-ref`, `svmc-jax`, and `svmc-js`.
3. **Invariant Validation**: JAX tests must include physical/metamorphic invariant checks (e.g., monotonicity, mass conservation) to expose plausible bugs that static fixture playback misses. TS-side invariant tests are recommended but not gating; the JAX invariants are the primary validation layer since they run at float64 and have autodiff access.
4. **Numerically Faithful & Differentiable**: JAX implementation matches Fortran mathematically and handles non-smooth operations via autodiff-friendly constructs (`jax.lax.cond`, soft-clamps).
5. **Gradients & OOD Validated**: Verify that `jax.grad` produces finite, stable gradients across both the baseline fixture input range **and** Out-of-Distribution (OOD) adversarial bounds to ensure L-BFGS inversion won't crash on unphysical edge cases.
6. **Browser Port Verified & Memory Safe**: The `@hamk-uas/jax-js-nonconsuming` port passes fixture validation against the same reference data as JAX. If a JAX construct does not map cleanly, implement and document a fallback. Every TS test must pass under `checkLeaks` (zero leaked arrays). All TS source must pass the jax-js ESLint plugin (`recommended` config; upgrade to `strict` once existing chain issues are resolved). Tolerances must be justified from numeric limits and algorithm structure: derive them from machine epsilon and critical-path operation depth, measure observed errors against fixtures, and use `atol` only for true near-zero noise floors. Do not introduce broad fixed tolerances just to clear tests.
7. **Temporal Rollout Vetted** _(Phases 3-5 only)_: Stateful submodels must pass a 100+ step sequential rollout fixture in default TS `float32` mode to prove accumulation errors do not diverge fatally from JAX/Fortran `float64` baselines. When TS `float64` mode is available, use it as the tighter browser-side parity reference.
8. **Reference Provenance Verified**: If fixture comparisons expose suspicious behavior, prove whether it comes from upstream SVMC, harness instrumentation, or the port itself before encoding the behavior as expected. Upstream artifacts or dead code must be documented with source references and, when appropriate, an issue draft in `issues/`.
9. **Phase-Exit Gate**: A submodel's phase is complete only when all criteria pass in CI (`branch:audit` + pytest + vitest). Do not begin the next phase until the gate is green.
10. **No Silent Shortcuts**: If any criterion above is partially met or intentionally deferred, document the gap explicitly in the phase status below with a rationale. Do not mark a criterion as complete when coverage is incomplete, evaluator logic is looser than the actual branch guard, or tolerances are still provisional.

## Phase 0: Project Foundation & Initial Porting (Completed)

- **Repo Setup**: Established monorepo (`svmc-ref`, `svmc-jax`, `svmc-js`).
- **CI Gates**: Set up GitHub Actions CI workflow to actually enforce the Phase-Exit cross-language validation rules (Pytest + Vitest), plus the PORT-BRANCH registry and waiver audit.
- **Fortran Harness**: Added logging to vendor SVMC Fortran code to capture inputs/outputs. Compiled reference harness to generate a single-source-of-truth JSONL dataset.
- **Fixture Pipeline**: Created Python scripts to parse JSONL into structured JSON fixture files per module.
- **Initial Module Porting**: Successfully ported core foundational modules (e.g., `viscosity_h2o`, `density_h2o`) to JAX and TypeScript and matched fixed precision outputs (`expectClose` accounting for float32/float64 limits).

## Phase 1: All Leaf Functions

Strictly porting the lowest-level, independent functions first to isolate bugs.

- **P-Hydro Leaves**: `ftemp_arrh`, `gammastar`, `ftemp_kphio`, `quadratic`, `scale_conductivity`. ✅ All ported to JAX + TS, fixture-tested. JAX invariant-tested (monotonicity, substitution, differentiability).
- **SpaFHy Leaves**: `e_sat`, `penman_monteith`, `soil_water_retention_curve`, `soil_hydraulic_conductivity`, `exponential_smooth_met`, `aerodynamics`. ✅ All ported to JAX + TS, fixture-tested (24 aerodynamics cases across 4 LAI × 3 wind speed × 2 parameter sets). JAX invariant-tested (Clausius-Clapeyron monotonicity, non-negativity, soil monotonicity, aerodynamic positivity).
- **Allocation Leaves**: `inputs_to_fractions` (conversion to AWENH pools). ✅ Ported to JAX + TS, fixture-tested. JAX invariant-tested (mass conservation, linearity, H-pool zero).

### Phase 1 DoD Status

| Criterion                             | Status                                                                                                                                                                                             |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fortran reference covered             | ✅ 496 fixture cases across phydro (298), water (187), yasso (5). All leaf functions have harness coverage.                                                                                        |
| Fixture & state contract              | ✅ JSON fixtures shared across svmc-ref, svmc-jax, svmc-js                                                                                                                                         |
| Invariant validation                  | ✅ 20 invariant/metamorphic tests in JAX (monotonicity, conservation, differentiability, OOD gradients). TS-side has fixture-only tests — invariants are JAX-only by design (see DoD criterion 3). |
| Numerically faithful & differentiable | ✅ JAX uses jnp.where for branch-free autodiff                                                                                                                                                     |
| Gradients & OOD validated             | ✅ calc_kmm gradient, fn_profit differentiability, fn_profit OOD (tc=45, patm=60k, vpd=5k), quadratic differentiability, soil_retention differentiability                                          |
| Browser port verified & memory safe   | ✅ 496/496 vitest pass with checkLeaks, ESLint clean                                                                                                                                               |
| Phase-exit gate                       | ✅ branch audit + pytest (516) + vitest (496) all green.                                                                                                                                           |

### Phase 1 Known Shortcuts

- **`exponential_smooth_met` Fortran harness copy**: The Fortran harness duplicates this routine (to avoid the yasso20 dependency chain). Constants α₁/α₂ are centralized in `packages/svmc-ref/constants/water.json`; the harness hardcodes matching values with a comment pointing to the canonical source.
- **TS constant allocation**: `inputs-to-fractions.ts` converts JS arrays to `np.array()` on every call; `quadratic.ts` allocates multiple zero-scalar arrays per call. Inside `jit()` these are constant-folded (traced once, baked into compiled program). The eager path pays per-call allocation cost — acceptable for non-hot-loop usage, to be revisited when wrapping the simulation loop in `jit()`.
- ~~**Harness IO Brittle stdout**~~: ✅ Fixed — harness now writes to `fixtures.jsonl` via Fortran file unit 10; `generate.py` reads from the file instead of parsing stdout.
- ~~**Disjointed Python venvs**~~: ✅ Fixed — `pyproject.toml` hoisted to repo root with `[tool.pytest.ini_options] testpaths`; CI uses single `pip install -e .[dev]` from root. Old `packages/svmc-jax/.venv/` is obsolete.

## Phase 2: Intermediate P-Hydro Assemblies & Optimizer

Combine the confirmed leaf functions into their dependent wrappers.

- `calc_kmm` (depends on `ftemp_arrh`). ✅ JAX + TS ported, fixture-tested.
- `calc_assim_light_limited` (depends on `quadratic`). ✅ JAX + TS ported, fixture-tested.
- `calc_gs` (depends on `scale_conductivity`). ✅ JAX + TS ported, fixture-tested.
- `fn_profit` (objective function forming the core of P-Hydro optimization). ✅ JAX + TS ported, fixture-tested, OOD gradient-tested.
- **Optimizer Overhaul (`optimise_midterm_multi`)**: ✅ Both JAX and TS now use projected Optax Adam (512 steps, lr=0.05, grad-clipping=10) with traced autodiff (`jax.value_and_grad` / `valueAndGrad`), replacing both Fortran's finite-difference L-BFGS-B and the earlier scipy/custom-L-BFGS approaches. The entire optimizer runs inside `jit`/`lax.scan` (TS) or `jax.lax.fori_loop` (JAX), keeping the optimization composable with larger JIT-compiled loops. Both fixture-tested against 6 reference cases.
  - _JAX vs Fortran tolerance: ~0.3% relative. TS default `float32` tolerance: ~5% relative. TS `float64` mode is available via `SVMC_JS_DTYPE=float64` for tighter browser-side parity checks._
  - _Invariant-tested: VPD monotonicity, drought monotonicity, aj/gs/ci consistency._
- `pmodel_hydraulics_numerical` (the overarching solver wrapper). ✅ JAX + TS ported, 7 fixture reference cases spanning environmental gradients, invariant-tested.

### Phase 2 DoD Status

| Criterion                             | Status                                                                                                                                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Fortran reference covered             | ✅ 65 fixture cases across calc_kmm (13), calc_gs (13), calc_assim_light_limited (13), fn_profit (19), pmodel_hydraulics_numerical (7).                                                                            |
| Fixture & state contract              | ✅ JSON fixtures shared across svmc-ref, svmc-jax, svmc-js                                                                                                                                                         |
| Invariant validation                  | ✅ 7 invariant/metamorphic tests in JAX (calc_kmm gradient, fn_profit differentiability + OOD gradients, solver profit positivity, VPD monotonicity, drought monotonicity, aj/gs/ci consistency).                   |
| Numerically faithful & differentiable | ✅ Both JAX and TS use projected Optax Adam with traced autodiff; entire optimizer runs inside `jit`/`lax.scan` (TS) or `jax.lax.fori_loop` (JAX).                                                                 |
| Gradients & OOD validated             | ✅ fn_profit OOD (tc=45, patm=60k, vpd=5k), calc_kmm gradient finite, fn_profit differentiable                                                                                                                    |
| Browser port verified & memory safe   | ✅ 65/65 vitest pass with checkLeaks (zero leaked arrays), ESLint jax-js `recommended` config applied                                                                                                              |
| Phase-exit gate                       | ✅ pytest (526) + vitest (503) all green.                                                                                                                                                                           |

### Phase 2 Known Shortcuts

- **Optax replaces scipy/Fortran L-BFGS-B**: The JAX solver uses `optax.adam` inside `jax.lax.fori_loop` instead of `scipy.optimize.minimize`. The TS solver uses `lax.scan` with Optax `adam` + `clipByGlobalNorm`. Both match Fortran reference outputs within tolerance but use a fundamentally different optimizer algorithm (first-order Adam vs quasi-Newton L-BFGS-B).
- **TS precision modes**: `svmc-js` now supports `SVMC_JS_DTYPE=float32|float64`. Solver tests and default browser runs still target `float32` because that is the primary performance mode; `float64` is available for higher-accuracy parity checks and future rollout comparisons.

## Phase 3: SpaFHy Submodels (Canopy & Soil Water Balance)

Move to canopy/soil hydrology processes which manage the local water states.

- Intermediate modules: `canopy_water_snow` and `ground_evaporation` (both depend on `penman_monteith`). ✅ JAX + TS ported, fixture-tested.
- Core Wrappers: `canopy_water_flux` and `soil_water`. ✅ JAX + TS ported, fixture-tested.

### Phase 3 DoD Status

| Criterion                             | Status                                                                                                                                                                                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fortran reference covered             | ✅ 29 fixture cases across ground_evaporation (8), canopy_water_snow (8), canopy_water_flux (8), soil_water (5). Branch-triggering fixtures now cover warm/cold/mixed precipitation, sublimation vs evaporation vs precip-suppressed no-evap, positive snow melt/freeze/no-change, LAI=0 guard, and saturation. 7 Phase 3 `PORT-BRANCH` ids are registered in vendor + registry + evaluator; branch audit passes (24 total, 20/24 covered, 4 pre-existing waivers). |
| Fixture & state contract              | ✅ JSON fixtures shared across svmc-ref, svmc-jax, svmc-js (water.json, 220 total records).                                                                                                                                                     |
| Invariant validation                  | ✅ 7 invariant/metamorphic tests in JAX: CWS mass balance, soil water mass balance, ground evap non-negative, ground evap zero-with-snow, CWS differentiable, soil water differentiable, canopy precip monotonicity.                              |
| Numerically faithful & differentiable | ✅ All `jnp.where` branch-free for AD; soil_hydraulic_conductivity uses `jnp.where(not_saturated, k_formula(safe=0.5), ksat)` for gradient-safe saturation handling.                                                                             |
| Gradients & OOD validated             | ✅ canopy_water_snow_differentiable and soil_water_differentiable tests verify finite, non-zero gradients through the full functions.                                                                                                             |
| Browser port verified & memory safe   | ✅ 29/29 Phase 3 vitest cases pass with checkLeaks (zero leaked slots). All operation chains broken into explicit `using` intermediates; `np.logicalAnd` replaced with `.mul()` (nonconsuming). `svmc-js` supports both `float32` and `float64` via immutable load-time configuration from `SVMC_JS_DTYPE`. |
| Phase-exit gate                       | ✅ branch audit (24 tags, 20/24 covered) + pytest (562) + vitest (535) all green. Full `svmc-js` suite passes in both `SVMC_JS_DTYPE=float32` and `SVMC_JS_DTYPE=float64` modes. |

### Phase 3 Known Shortcuts

- **Fortran mbe accounting artifact** (`vendor/SVMC/src/water_mod.f90` L128 `rr = potinf + PondSto`, L185-187 mbe formula): The Fortran `soil_water` mass-balance formula uses `rr = potinf + PondSto_old`, double-counting `PondSto_old` in the flux term. JAX and TS replicate this formula exactly; tests compare against fixture expected mbe rather than asserting mbe ≈ 0.
- **Lateral flow hardcoded to 0** (`vendor/SVMC/src/water_mod.f90` L118 `latflow = 0.0`, L145-146 re-zeroed after comment `! lateral drainage to ditches here!`): The Fortran reference hardcodes lateral drainage to zero. Both JAX and TS replicate this; the parameter is reserved for future use.
- **ET and Transpiration set to 0 in `canopy_water_flux`**: These will be populated by P-Hydro in Phase 5 integration.
- **TS precision switch**: `svmc-js` array creation is routed through a project-local precision wrapper. Default mode is `float32`; `SVMC_JS_DTYPE=float64` enables higher-accuracy execution without changing call sites. The dtype is fixed at module-load time rather than mutated globally at runtime.
- **TS float32 tolerances**: All TS test tolerances are derived from machine epsilon (`baseNp.finfo(dtype).eps`) scaled by each function's critical-path operation depth with a ≥2× safety factor.  Near-zero fields (mbe, Kh) use an `atol` noise floor instead of loose relative tolerance — `128 * eps` for mass-balance sums, `eps` for Mualem-formula underflow.  The `sublim_vs_evap` branch-audit evaluator additionally requires `LAI > eps` to match the enclosing Fortran guard.

## Phase 4: Carbon Allocation & Yasso20 Decomposition

Port the long-term, daily/yearly loop processes.

Quality bar carried over from Phase 3:

- Add explicit branch-trigger fixtures for each biologically distinct allocation and decomposition regime instead of relying on seasonal baseline runs alone.
- Treat long-horizon state rollouts as first-class validation artifacts: compare not only endpoint state but intermediate pool trajectories and conservation residuals.
- If Yasso or allocation reference behavior appears inconsistent, verify provenance against upstream SVMC before preserving it in fixtures or tolerances.
- Any TS tolerance for long-horizon rollouts must be backed by measured drift and epsilon-scaled accumulation analysis, not a blanket per-phase relaxation.

- **Fortran Build Modernization (`fpm`/`CMake`)**: At this point, integrating the 20+ Yasso and Allocation sub-modules inside the Fortran harness will break the manual `Makefile` dependency ordering. Before writing Yasso harness logs, scrap the `Makefile` and adopt `fpm` (Fortran Package Manager) or `CMake` for reliable automated dependency parsing. Doing this early in Phase 4 defers scope creep but prevents a bottleneck precisely when the Fortran inclusion graph becomes intractable.
- Allocation modules: `invert_alloc` and `alloc_hypothesis_2`.
- Yasso20 Soil Carbon:
  - `yasso.initialize_totc` (Setup routines).
  - `yasso.decompose` (Daily decomposition).
  - `yasso20.mod5c20` (Yearly spin-up calculations; matrix exponentials).
  - _TS Port Note: `jax-js-nonconsuming` currently lacks `expm`. Request the `jax-js-nonconsuming` team to implement it before the Yasso20 TS port begins._

## Phase 5: Main SVMC Integration Loop

Combine the differentiable submodels into the top-level time-step loops.

Quality bar carried over from Phase 3:

- Preserve branch-audit discipline at integration level: when a wrapper introduces new conditional behavior, add `PORT-BRANCH` tags and evaluator logic that matches the real guard conditions exactly.
- Validate coupled rollouts with multi-step fixtures that force regime switches, not just steady-state weather segments.
- Keep known upstream accounting artifacts separate from true port regressions so integration tests do not normalize new bugs.

- **I/O Boundary Rule**: Keep namelist parsing and netCDF reading in a thin adapter layer separate from the computational core. Submodel functions must accept plain arrays/scalars so they remain testable and composable outside the original SVMC file conventions.
- Wire up initializers via the adapter layer: Reading namelists (Configs) and starting `initialization_spafhy` & `wrapper_yasso_initialize_totc`.
- Construct the **Hourly Loop** using JAX loops (`jax.lax.scan` or `jax.lax.fori_loop`) coupling `P-Hydro` → `canopy_water_flux` → `soil_water`.
- Construct the **Daily/Yearly Loops** wrapping allocations and Yasso updates.
- Verify overall system behavior against original Fortran integrated outputs.

## Phase 6: Interactive Web Application

Once the full pipeline passes in `jax-js-nonconsuming` (`svmc-js`), build the front-end to utilize the fast browser-based execution.

- Initialize a web-app via Vite/React (or preferred framework).
- Map configurable namelists to a UI parameters dashboard.
- Expose the `P-Hydro`, `SpaFHy`, and `Yasso` states as reactive charts.
