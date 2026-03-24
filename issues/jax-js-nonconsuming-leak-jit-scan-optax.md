# 62 slots leaked internally by `jit(fn)` wrapping `lax.scan` + `valueAndGrad` + Optax

**Library**: `@hamk-uas/jax-js-nonconsuming` v0.9.3 (installed from `github:hamk-uas/jax-js-nonconsuming`)  
**Test runner**: Vitest 3.2.4, browser mode (chromium / Playwright)  
**Repo**: This repo — `packages/svmc-js/`  
**Reproducer**: `pnpm exec vitest run packages/svmc-js/test/phydro-leaf.test.ts`

## Summary

After a call to a short-lived `jit(impl)` where `impl` contains `lax.scan` with
inline `valueAndGrad` and Optax optimizer steps, `checkLeaks` reports **62 leaked
slots (130 tracked arrays)** per call. Of these, **129 are attributed to
`.vite`-bundled library code** and **1 to user code**. The single user-attributed
leak shifts to whichever array allocation happens to be first after the optimizer
returns, suggesting it is a library-internal attribution artifact. All arrays have
`rc=1` (never disposed).

The leak count is **exactly 62 slots every time**, regardless of input values or
which test case triggers it. It happens on every call that goes through the
`jit` → `scan` → `valueAndGrad` → Optax path.

## What our code does

The triggering code is in [`packages/svmc-js/src/phydro/solver.ts`](../packages/svmc-js/src/phydro/solver.ts).

### Pattern

```ts
import { clearCaches, jit, lax, numpy as np, tree, valueAndGrad } from "@hamk-uas/jax-js-nonconsuming";
import { adam, applyUpdates, chain, clipByGlobalNorm, type OptState } from "@hamk-uas/jax-js-nonconsuming/optax";

const MAX_OPT_STEPS = 512;
const OPTIMIZER = chain(clipByGlobalNorm(10.0), adam(0.05));

function optimiseMidtermMultiImpl(
  /* 18 flat np.Array scalar args */
): OptimResult {
  // Reconstruct param structs from flat args
  // ...

  // Objective closure (uses using for intermediates):
  const objective = (params, ...closedOverArgs) => {
    using logJmax = lax.dynamicIndexInDim(params, 0, 0, false);
    using dpsi    = lax.dynamicIndexInDim(params, 1, 0, false);
    return fnProfit(logJmax, dpsi, ...closedOverArgs);
  };

  using initParams = np.array([4.0, 1.0]);
  const initCarry = {
    params: initParams,
    optState: OPTIMIZER.init(initParams),
    bestParams: np.array([4.0, 1.0]),
    bestLoss: np.array(Infinity),
  };

  // Scan-based optimizer loop
  const step = (carry: OptimCarry, _: null): [OptimCarry, null] => {
    const [loss, grads] = valueAndGrad(objective, { argnums: 0 })(
      carry.params, ...closedOverArgs,
    );
    const better = loss.less(carry.bestLoss);
    const bestParams = np.where(better, carry.params, carry.bestParams);
    const bestLoss   = np.where(better, loss, carry.bestLoss);
    const [updates, optState] = OPTIMIZER.update(grads, carry.optState, carry.params);
    const params = projectParams(applyUpdates(carry.params, updates));
    return [{ params, optState, bestParams, bestLoss }, null];
  };

  const [finalCarry] = lax.scan(step, initCarry, null, { length: MAX_OPT_STEPS });

  // Extract results, clone out of carry, dispose carry
  using finalLoss = objective(finalCarry.params, ...);
  using better = finalLoss.less(finalCarry.bestLoss);
  using bestParams = np.where(better, finalCarry.params, finalCarry.bestParams);
  // ... extract jmax, dpsi, objectiveLoss via .add(0) ...
  tree.dispose(finalCarry);
  return { jmax, dpsi, objectiveLoss };
}

// Short-lived jit wrapper
function optimiseMidtermMultiFlat(...args): OptimResult {
  using core = jit(optimiseMidtermMultiImpl);
  const result = core(...args);
  clearCaches();
  return result;
}
```

### Caller (`pmodelHydraulicsNumerical`)

The caller builds 18 scalar `np.Array` inputs in an IIFE scope, calls
`optimiseMidtermMultiFlat`, manually disposes all 18 inputs, then constructs
fresh diagnostic param structs via `tree.makeDisposable` for post-optimizer
evaluation. All diagnostic arrays are disposed via `using`.

## Leak report (identical for all 7 solver tests)

```
62 slot(s) leaked (130 tracked array(s)):

  src/phydro/solver.ts:372:47 — 1× Array:float32[] rc=1  (via array → full)
  …/chunk-KLD7MJHE.js:1339:30 — 58× Array:float32[] rc=1, …  (.vite)
  …/chunk-KLD7MJHE.js:3793:39 — 40× Array:float32[] rc=1, Array:bool[] rc=1, …  (.vite)
  …/chunk-KLD7MJHE.js:2757:41 — 22× Array:float32[] rc=1, …  (.vite)
  …/chunk-KLD7MJHE.js:1340:19 — 8× Array:float32[] rc=1, …   (.vite)
  …/chunk-KLD7MJHE.js:1227:12 — 1× Array:float32[] rc=1      (.vite)

129 in .vite, 1 from user code.

Some tracked arrays share backend storage (130 tracked, 62 slots).
rc=1 → never disposed.
Package-tagged leaks are library bugs, not user error.
```

The user-attributed line (`solver.ts:372`) points to the first `np.array()`
call in the diagnostic section **after** the optimizer returns — it is not
related to the optimizer itself. If diagnostic code is restructured, this
attribution shifts to whichever allocation happens to be next.

## What we tried

1. **`lax.foriLoop` instead of `lax.scan`** — crashed with `Referenced tracer has been disposed`
2. **Persistent module-level `jit()` / `valueAndGrad()`** — increased leak count
3. **Short-lived `jit()` + `clearCaches()`** — current approach, same 62-slot leak
4. **`tree.makeDisposable` for all param structs** — no change
5. **Splitting optimizer inputs from diagnostic inputs** (separate construction scopes with manual dispose) — reduced user-attributed leaks from many to 1, but library-internal 129 unchanged
6. **`tree.dispose(finalCarry)`** — already done, no effect on the 129

## Expectation

After `tree.dispose(finalCarry)`, disposal of the `jit`-compiled function via
`using core`, and `clearCaches()`, we expect zero leaked slots. The 62 slots
(129 library-attributed arrays) appear to be retained internally by the `jit` /
`scan` / `valueAndGrad` / Optax machinery and are not reachable for user-side
disposal.

## Test command

```sh
pnpm exec vitest run packages/svmc-js/test/phydro-leaf.test.ts
```

All 7 tests in the "P-Hydro solver — Fortran reference" describe block fail
on the `checkLeaks` assertion in `packages/svmc-js/test/setup.ts`:

```ts
afterEach(() => {
  const result = checkLeaks.stop();
  expect(result.leaked, result.summary).toBe(0);
});
```

The other 300 tests (leaf functions that don't use `jit`/`scan`/`valueAndGrad`)
pass cleanly with 0 leaks.
