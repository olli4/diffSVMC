# 004 — PE intermediates leak in nested `grad`/`hessian` inside traced `scan`/`foriLoop` bodies

## Summary

After the v0.12.6 and v0.12.7 ownership fixes, the original traced-jaxpr
constant leak and disposed-tracer failures are largely resolved. However,
there is still a remaining leak path when nested autodiff runs inside traced
control-flow bodies:

- `lax.scan(...)` → `lax.foriLoop(...)` → `grad(...)`
- `lax.scan(...)` → `lax.foriLoop(...)` → `hessian(...)`

The remaining leaked slots appear to be **PE intermediates created during
abstract tracing**, not the old cached-jaxpr const ownership bug.

## Current impact in diffSVMC

The official v0.12.10 release passes the focused downstream suite in diffSVMC,
but one downstream workaround still remains because of this leak path:

1. `integration.test.ts` temporarily suspends leak assertions for the
  full 1-day `runIntegrationScanExperimental(...)` parity test and logs a
  known limitation:

  - `44 slot(s) leaked (PE intermediates in nested grad/hessian inside traced bodies)`

The previously skipped targeted solver diagnostic in `phydro-leaf.test.ts`
now passes on `v0.12.10`, so the remaining problem appears narrower than it
was on `v0.12.8`.

This means correctness is good enough to keep the main suites enabled, but
ownership is still not fully closed for nested autodiff in traced loops.

## Status update (2026-04-03)

This issue is no longer an active downstream blocker in `diffSVMC`.

After a downstream refactor that extracted the daily
allocation/decomposition step into a helper shared by the eager and scan paths,
we re-ran the two historical reproductions:

- `packages/svmc-js/test/integration.test.ts -t "scan experimental matches eager replay for 1 day"`
- `packages/svmc-js/test/phydro-leaf.test.ts -t "optimiseMidtermMulti works inside lax.scan when params are arrays"`

Current downstream result:

- the scan parity test now passes without emitting the historical leak warning
- the targeted solver-in-scan diagnostic also passes cleanly

So from the `diffSVMC` side, we do not currently have a live reproducer for
this issue. The notes below remain useful as historical context, but the
remaining eager vs outer-jitted integration work is now a downstream
architecture task rather than an active `jax-js-nonconsuming` bug report.

## Representative reproductions

### 1. Full integration parity path

Downstream location:

- `packages/svmc-js/test/integration.test.ts`

Pattern:

```ts
runIntegrationScanExperimental(inputs)
// internally: scan -> foriLoop -> pmodelHydraulicsNumerical
// solver path uses grad(...) and hessian(...)
```

Observed result on v0.12.10:

```txt
[known limitation] scan experimental: 44 slot(s) leaked
(PE intermediates in nested grad/hessian inside traced bodies)
```

### 2. Targeted solver reproduction

Downstream location:

- `packages/svmc-js/test/phydro-leaf.test.ts`

Pattern:

```ts
const [carry, ys] = lax.scan((c: np.Array, _x: np.Array): [np.Array, np.Array] => {
  const result = optimiseMidtermMulti(psiSoil, parCost, parPhotosynth, parPlant, parEnv);
  return [c, result.jmax];
}, init, xs, { length: 1 });
```

Observed result on v0.12.8 when enabled:

```txt
18 slot(s) leaked (52 tracked array(s))

  src/precision.ts:34:19 — 3× Array:float32[2] rc=1
  src/phydro/solver.ts:477:15 — 3× Array:float32[4,2] rc=1
  src/phydro/solver.ts:478:15 — 3× Array:float32[4,2] rc=1
  src/phydro/solver.ts:479:17 — 3× Array:float32[4] rc=1
  src/phydro/solver.ts:480:19 — 3× Array:bool[4] rc=1
  src/phydro/solver.ts:581:51 — 3× Array:float32[2,2] rc=1
  src/phydro/ftemp-kphio.ts:70:17 — 1× Array:float32[] rc=1
  test/phydro-leaf.test.ts:510:32 — 1× Array:float32[] rc=1
  ...plus library-internal `.vite` entries
```

Observed result on v0.12.10:

```txt
test passes cleanly (no leak-check failure)
```

## Why this looks different from issue 003

Issue 003 was about traced jaxpr constants and cached ownership lifetimes.
The v0.12.6 and v0.12.7 fixes addressed the dominant failures there:

- cached-jaxpr const ownership leak fixed in v0.12.6
- disposed-tracer / identity-shared value bug fixed in v0.12.7

What remains now appears narrower:

- correctness succeeds
- the old disposed-tracer crash is gone in the scan reproducer tests
- leaks remain specifically when PE/autodiff intermediates are created inside
  abstract traces under `scan`/`foriLoop`

## Suspected root cause

The downstream tests currently assume the remaining issue is in PE cleanup:

- `disposePeIntermediates` appears to skip cleanup inside abstract traces
- nested `grad(...)` / `hessian(...)` inside traced control-flow bodies leaves
  residual intermediates alive after the compiled function returns

This is only a hypothesis from downstream observation, but it matches the
current symptoms better than the old const-ownership explanation.

## v0.12.10 recheck

We re-ran the downstream cases after updating to the published
`@hamk-uas/jax-js-nonconsuming` `v0.12.10` release.

Observed downstream result:

- the full scan integration parity test still emits a known limitation, but it
  improved from `62` leaked slots on v0.12.8 to `44` leaked slots on v0.12.10:
  `44 slot(s) leaked (PE intermediates in nested grad/hessian inside traced bodies)`
- the targeted `optimiseMidtermMulti works inside lax.scan when params are arrays`
  diagnostic now passes cleanly on v0.12.10

So from the downstream `diffSVMC` point of view, `v0.12.10` fixes the targeted
solver reproduction and reduces the remaining scan-parity leak, but does not
yet remove the remaining leak path tracked in this issue.

That statement is now stale for current `diffSVMC` HEAD: the downstream shared
helper refactor above removed the remaining repros in this repository.

## Desired fix

The library should ensure PE intermediates created during nested autodiff
inside traced control-flow bodies are released symmetrically when the trace
completes.

Concretely, downstream would like to be able to:

1. run `lax.scan` / `lax.foriLoop` bodies that call `grad(...)` / `hessian(...)`
   without special-case leak-check suppression;
2. keep the `optimiseMidtermMulti(...) inside lax.scan` regression test enabled;
3. drop the test-local leak-check workaround from the scan integration parity
   test.

## Notes for follow-up / docs

If this behavior is expected for now rather than a bug, it would help to
document it explicitly in upstream ownership guidance for traced control flow
and nested autodiff, because downstream projects otherwise read the current
state as “all ownership issues are fixed” once v0.12.10 is installed.