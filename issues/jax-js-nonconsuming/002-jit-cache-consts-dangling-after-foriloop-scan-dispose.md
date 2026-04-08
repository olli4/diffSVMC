# `jit()` cache consts become dangling after outer `foriLoop`/`scan` disposes shared ClosedJaxpr

## Component

`@hamk-uas/jax-js-nonconsuming` — `jit$1()` in `dist/index.js`

## Problem

When a `jit()`-wrapped function is called inside a traced scope
(`foriLoop` or `scan` body), the jit cache stores a `ClosedJaxpr` whose
`consts` are shared `Array` objects with the enclosing primitive's
`ClosedJaxpr`. When the outer primitive disposes its `ClosedJaxpr`, the
shared consts are freed to refcount 0. The jit cache is left holding
dead references, and calling `_disposeAllJitCaches()` (via
`checkLeaks.stop()`) triggers a `UseAfterFreeError`.

### Reproduction lifecycle

```
1. foriLoop(body) traces body via makeJaxpr(body)
2. body calls jit(f)(x) — jit caches ClosedJaxpr-A with const arrays
   (e.g. np.eye(2))
3. jit passes ClosedJaxpr-A.consts into outer trace via bind() →
   getOrMakeConstTracer → .ref → rc bumped temporarily
4. Outer makeJaxpr finishes → constsNeedingCreationRefBalance balances
   the creation ref → rc reduced back
5. foriLoop calls closedJaxpr.dispose() → shared consts freed to rc=0
6. jit cache still holds ClosedJaxpr-A with dead const references
7. _disposeAllJitCaches → ClosedJaxpr-A.dispose() → UseAfterFreeError
```

### Minimal reproducer (conceptual)

```ts
const f = jit((x: np.Array) => {
  using eye = np.eye(2);
  return np.dot(eye, x);
});

// foriLoop traces f — jit caches the ClosedJaxpr containing np.eye(2)
// After foriLoop disposes its ClosedJaxpr, the cached eye is dangling
const result = foriLoop(0, 10, (i, x) => f(x), np.zeros([2]));
```

### Root cause

`runWithCache` (in `utils-DDTPOEK7.js`) is a simple `Map` cache keyed
by JSON-serialised args. On cache hit it returns the same `ClosedJaxpr`
object — no copy, no additional refcount on consts. Nothing prevents the
outer scope from disposing the consts out from under the cache.

Relevant code locations (line numbers from `dist/index.js`):

| Location | Line(s) | Role |
|---|---|---|
| `jit$1()` | ~3030–3080 | Creates cache via `runWithCache`, stores disposer in `_jitFunctionDisposers` |
| `runWithCache` | utils ~437 | Simple Map cache, returns same object on hit |
| `ClosedJaxpr.dispose()` | ~2482 | Iterates `this.consts`, calls `.dispose()` on each |
| `getOrMakeConstTracer` | ~2525 | Does `tval.ref` but balanced away by `constsNeedingCreationRefBalance` |
| `foriLoop` | ~20793 | Calls `makeJaxpr(body)` then `closedJaxpr.dispose()` |

## Resolution

**Fixed in v0.12.6** — Two-layer ownership model via `retainClosedJaxpr()`.
The cache now holds: (1) a retained `ClosedJaxpr` (with `.ref`'d const copies)
for execution, and (2) the original traced `ClosedJaxpr` as transient owner
of builder-held refs. Both are released at `jit.dispose()` / `clearCaches()`,
with `UseAfterFreeError` tolerance for consts already freed by enclosing scopes.
| `_disposeAllJitCaches` | ~54 | Iterates all jit disposers — triggers the crash |

## Suggested fix

Take an additional `.ref` on each const when storing the `ClosedJaxpr`
in the jit cache, so the cache holds its own refcount independently of
any outer scope:

```js
// In jit$1(), after caching the ClosedJaxpr:
for (const c of jaxpr.consts) c.ref;  // bump rc for cache ownership
```

This ensures the cache's consts survive disposal by outer primitives.
The existing `result.dispose` callback (registered in
`_jitFunctionDisposers`) already disposes the cached `ClosedJaxpr`
entries, which will balance the extra ref.

## Workaround

Do not wrap functions in `jit()` if they are called from inside traced
scopes (`foriLoop`, `scan`, `whileLoop` bodies). Instead call them
directly — the outer `makeJaxpr` trace already absorbs the function's
operations.

This is what we applied in `svmc-js` (`packages/svmc-js/src/phydro/solver.ts`)
by removing the `jit()` wrapper from `projectedNewtonSolveImpl`.

## Context

Discovered while debugging "Referenced tracer Array has been disposed"
errors in the P-Hydro Newton/L-BFGS solver port. The solver's
`projectedNewtonSolve` was wrapped in `jit()` and called inside
`foriLoop` (iterative optimisation loop). After removing the `jit()`
wrapper, the disposal errors stopped.
