---
description: "Use when porting SVMC submodels, adding Fortran reference logging, generating fixtures, implementing JAX modules, porting to jax-js-nonconsuming, or building interactive browser demos for validated submodels."
name: "SVMC Conservative Porting Workflow"
---
# SVMC Conservative Porting Workflow

- Treat the maintained SVMC Fortran reference tree in `vendor/SVMC/` as the reference source of truth for numerical behavior. It derives from `huitang-earth/SVMC` but may include repo-local, non-numerical porting aids.
- Work bottom-up from the leaf functions in DEPENDENCY-TREE.md and follow the phase order in PLAN.md unless the user explicitly reprioritizes.
- Before porting a submodel, add or extend reference logging in `packages/svmc-ref` so the repo captures representative inputs and outputs from the original model. For wrappers, log at the wrapper boundary to catch integration issues.
- Use the example input data included in the repo as the baseline reference run. For robust coverage, add targeted single and combined branch-triggering cases whenever a submodel has input-dependent logic.
- Generate fixture files from the Fortran harness and treat their schemas as strict cross-package interface contracts for JAX and TypeScript.
- Augment fixture-playback tests with metamorphic/invariant tests (e.g., monotonicity, conservation laws) to catch issues just outside the logged paths.
- Implement the submodel in JAX first. Keep the implementation differentiable and suitable for gradient-based parameter tuning in downstream inversion workflows.
- After the JAX version matches the reference fixtures, port the same validated behavior to `@hamk-uas/jax-js-nonconsuming` in `packages/svmc-js` and verify the TypeScript output against the same fixtures.
- Account for precision differences explicitly: JAX should match the reference as closely as practical, while `svmc-js` tests may need documented tolerances consistent with `jax-js-nonconsuming` numeric limits.
- If a JAX construct lacks direct support in `jax-js-nonconsuming`, explicitly document and implement a standardized fallback policy rather than ad hoc approximations.
- Do not move on to higher-level composition until the current leaf or submodel is validated in both JAX and TypeScript against reference data.
- Only build or expand the interactive website for a submodel after the `svmc-js` port of that submodel is already verified.
- When adding tests, prefer fixture-driven combined with invariant-driven coverage over hand-written constants. Require explicit branch-triggering fixtures for input-dependent branches, documenting the exercised conditions in the test data or names.