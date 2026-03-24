# Contributing: PORT-BRANCH Convention

## What It Enforces

Every conditional branch in the vendor Fortran must be:

1. Annotated in source with a `PORT-BRANCH` tag.
2. Registered in `packages/svmc-ref/branch-coverage.json`.
3. Audited by `scripts/verify_branch_coverage.py`.
4. Either fully covered on both fixture sides or explicitly waived.

CI now runs the branch audit, so this is no longer documentation-only discipline.

## Annotation Format

In vendor Fortran (`vendor/SVMC/src/*.f90`) and in `packages/svmc-ref/harness.f90`,
every branch gets exactly two comment lines immediately before the branch point:

```fortran
! PORT-BRANCH: domain.function.branch_name
! Condition: plain-language description of what triggers this branch
if (some_condition) then
```

For `min` or `max` clamps that behave as implicit branches:

```fortran
! PORT-BRANCH: water.soil_retention.porosity_floor
! Condition: watsat - vol_ice < 0.01 -> clamp effective porosity to 0.01
eff_porosity = max(0.01, watsat - vol_ice)
```

If the same logic is duplicated in `packages/svmc-ref/harness.f90`, annotate the harness copy too. The harness tag must reuse the same branch id.

## Branch Ids

Ids follow `domain.function.semantic_name` and must be unique across the repository.

Examples:

- `phydro.quadratic.linear_fallback`
- `water.aerodynamics.rb_lai_guard`
- `yasso.exponential_smooth_met.init_vs_smooth`

## Registry Format

All authoritative branch metadata lives in `packages/svmc-ref/branch-coverage.json`.

Each branch entry records:

- `id`: exact match for the `PORT-BRANCH` tag.
- `file`: relative vendor file containing the authoritative tag.
- `condition`: what makes the branch condition true.
- `else_behavior`: what happens on the opposite side.
- `fixture_both_sides`: boolean computed by the audit script. Do not guess this.
- `jax_tested`: whether the JAX port has coverage for the registered behavior.
- `ts_tested`: whether the TypeScript port has coverage for the registered behavior.
- `notes`: human explanation of the current fixture status.
- `waiver`: required whenever `fixture_both_sides` is `false`.

Waivers must include:

- `scope`
- `kind`
- `approved_by`
- `reason`

## Waiver Kinds

Uncovered branches are not all the same. Use one of these typed waiver categories so similar cases are handled consistently:

- `fixture-gap`: the branch is reachable and meaningful, but the harness does not yet emit a valid reference trigger case.
- `dead-code`: the branch is structurally unreachable under the current upstream control flow or invariants.
- `fatal-path`: the branch intentionally aborts execution, so it needs isolated error-path testing rather than normal fixture playback.
- `undefined-behavior`: the branch returns undefined or unstable state, so it must not be fixture-validated until the reference behavior is stabilized.

Each waiver kind requires additional evidence:

- `fixture-gap` requires `next_action`.
- `dead-code` requires `evidence`.
- `fatal-path` requires `safe_test_strategy`.
- `undefined-behavior` requires `stabilization_plan`.

Example:

```json
{
	"scope": "fixture-both-sides",
	"kind": "dead-code",
	"approved_by": "phase-1-baseline",
	"reason": "The later clamp cannot trigger because an earlier cap already bounds the same quantity.",
	"evidence": "zg1 = min(zground, 0.1*hc) implies zg1/hc <= 0.1, so min(zg1/hc, 1.0) never takes the capped side."
}
```

## Audit Command

Run the audit locally with:

```bash
pnpm branch:audit
```

or:

```bash
python3 scripts/verify_branch_coverage.py
```

The audit fails when:

- a vendor `PORT-BRANCH` tag is missing from the registry,
- a registry entry points at the wrong file,
- a tag is missing its `! Condition:` line,
- `fixture_both_sides` does not match the computed fixture truth,
- an uncovered branch has no waiver,
- a waiver is missing its kind-specific evidence field,
- a fully covered branch still carries a waiver,
- the registry summary is stale.

## Workflow For New Ported Functions

1. Identify every branch in the vendor Fortran.
2. Add `PORT-BRANCH` and `Condition` comments in the vendor source.
3. If the branch logic is duplicated in the harness, annotate the duplicate too.
4. Add a registry entry with the correct file, condition, else behavior, and notes.
5. Add fixture cases that exercise both sides when practical.
6. If both sides are still not covered, add an explicit typed waiver explaining which exception class applies and what the next safe action is.
7. Run `pnpm branch:audit` before considering the work complete.

## Exception Decision Rule

When a branch cannot be fully covered yet, classify it before writing the waiver:

1. If it is reachable and should eventually be fixture-covered, use `fixture-gap`.
2. If an earlier guard or invariant makes it unreachable, use `dead-code`.
3. If taking the branch aborts the program, use `fatal-path`.
4. If taking the branch returns undefined or unstable state, use `undefined-behavior`.

Do not use a generic explanation when one of these categories applies.

## Counting Rules

- `max(a, b)` and `min(a, b)` count as one branch.
- Each `if` condition counts as one branch.
- Nested conditions are counted separately.
- Registry coverage is branch coverage, not path-combination coverage.

## Branch-Free Functions

Functions with no conditional logic do not need `PORT-BRANCH` tags and are not listed in the registry.
