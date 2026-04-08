# 005 — eager shared-kernel leak bisects to inline `where(...)` branch temporaries

## Summary

We reduced the remaining eager/shared-kernel ownership problem to an upstream
`jax-js-nonconsuming` reproducer that no longer depends on `diffSVMC`
integration code, and the latest bisect changes the conclusion materially.

The failing boundary is:

- eager scalar state is materialized into arrays
- a reused traced kernel is called from eager code
- the kernel contains nested traced control flow and autodiff:
  - `foriLoop`
  - `grad(...)`
  - `hessian(...)`
  - `np.linalg.solve(...)`
  - boolean `np.where(...)`
- the eager wrapper scalarizes the returned carry and then performs
  alias-aware cleanup of inputs and outputs

The leaking pattern is now narrower:

- the leak still reproduces with `params + running total`
- but it disappears when the conditional `where(...)` branch temporaries are
  first bound with explicit `using` ownership

So this no longer looks like a generic upstream runtime leak in the reused
traced kernel itself. It currently looks like an ownership-discipline issue in
the eager wrapper pattern when inline branch arrays are created and passed
directly into `where(...)`.

## Upstream reproducer

Location:

- `jax-js-nonconsuming/test/eager-shared-kernel-repro.test.ts`

Validation command:

```bash
cd ../jax-js-nonconsuming
pnpm vitest run test/eager-shared-kernel-repro.test.ts
```

The upstream test is intentionally a characterization test for now.
It now demonstrates both sides of the boundary:

- leaking variants with inline conditional branch temps
- leak-free variants where those branch temps are explicitly owned/disposed

## Carry-field bisect result

Current reduced result from the upstream repro:

1. Full carry leaks.
2. Removing `prevObs` and `aux` does **not** remove the leak.
3. `params + running total` is still sufficient to reproduce the leak.
4. `params` alone is leak-free.
5. `params + running total` with unconditional update is leak-free.
6. `params + running total` with conditional update is also leak-free when the
  `where(...)` branch temps are bound explicitly via `using`.

So the minimal currently-known leaking boundary is not the whole integration
carry shape. It is present when the eager wrapper rematerializes just:

- the traced state vector (`params`)
- one running scalar accumulator (`total`)
- and updates that accumulator via inline `where(...)` branch expressions

while reusing the traced step closure.

## Why this matters for diffSVMC

This means the remaining blocker for the eager/shared-day-kernel merge is not
primarily downstream transfer bookkeeping on the full `DailyCarry` shape.

The downstream wrapper still needs alias-aware cleanup, but the latest upstream
reproducer suggests the remaining leak can likely be avoided downstream by
making conditional branch temps explicit rather than inlining them into
`where(...)`. That is why the earlier `integration.ts` production-path
experiment was reverted: it mixed transfer semantics and inline temporary
ownership, so it was not yet a safe merge.

## Relationship to issue 004

This looks related to the same general ownership surface as issue 004, but it
is a distinct boundary:

- issue 004 focused on leaks from nested `grad`/`hessian` inside traced
  `scan`/`foriLoop` bodies
- this issue focuses on eager code that repeatedly rematerializes scalar state,
  calls a reused traced kernel, and then reclaims materialized inputs/outputs

The new upstream reproducer suggests that this specific reduced case is not a
generic runtime ownership defect. It is closer to a `diffSVMC`-relevant
ownership pattern at the call site.

## Recommended downstream follow-up

Before reopening an upstream bug here, try the same ownership pattern in the
real eager/day-kernel wrapper:

1. pull inline `where(...)` branch expressions into named `using` temporaries
2. keep alias-aware cleanup for escaped carry/output leaves
3. re-run the focused integration parity path

## Concrete downstream note

When retrying the eager/shared-day-kernel merge in `packages/svmc-js/src/integration.ts`,
the reduced upstream result suggests treating this as a call-site ownership rule:

Bad pattern:

```ts
const nextTotal = np.where(better, nextParams.sum(), carry.total.add(obs));
```

Preferred pattern:

```ts
using nextParamsSum = nextParams.sum();
using fallbackTotal = carry.total.add(obs);
using better = nextParamsSum.greater(carry.total);
const nextTotal = np.where(better, nextParamsSum, fallbackTotal);
```

That pattern should be applied anywhere the eager wrapper feeds newly created
branch temporaries directly into `where(...)`, especially for carry leaves that
will survive past the local expression.

This note is intentionally narrow: it does **not** prove the full integration
merge will be leak-free by itself, but it removes the strongest reduced case
that previously looked like an upstream runtime ownership bug.

The concrete downstream goal remains the same:

- safely route eager per-day execution through the shared traced daily kernel
- without disposed-tracer faults
- and without per-day slot leaks

## Status update (2026-04-08)

**Closed — not applicable to current architecture.**

The eager/shared-day-kernel merge that this issue was analysing was never
implemented. The current `diffSVMC` architecture uses two independent
integration modes:

- **eager** (`runIntegration`): imperative JS loop, no tracing
- **scan** (`runIntegration`): fully traced via `lax.scan`

Both modes pass all tests cleanly on v0.12.14 with zero leaks, including the
scan parity test that previously required a `checkLeaks` workaround (removed
in the same session — see issue 004 resolution).

A local reproducer of the reduced eager-calling-grad pattern confirmed that
the inline `where()` branch-temp leak (7 slots inline vs 5 slots explicit
`using`) still exists on v0.12.14 in pure eager `grad` calls. However, this
pattern is not exercised anywhere in the current codebase — `grad` is only
called inside traced bodies (scan → foriLoop → solver).

If the shared-kernel merge is revisited in the future, the ownership rules
documented above (explicit `using` for `where()` branch temps) remain relevant.
For now, this issue has no active reproduction path in `diffSVMC`.
