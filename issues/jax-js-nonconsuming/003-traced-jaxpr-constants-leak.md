# 003 — Traced jaxpr constants leak (never disposed)

## Summary

When user code calls array-producing functions (`np.zeros`, `np.array`,
arithmetic operations like `.mul`, `.add`, `.sub`) inside a traced scope
(`lax.scan` body, `lax.foriLoop` body), the library creates concrete backend
arrays as jaxpr constants. These constants are captured in the compiled jaxpr
but are **never disposed** after the compiled function completes execution.

## Impact

In the full 1-day `runIntegration` integration test, this
causes **417 user-code-site array leaks** (plus ~147 000 library-internal
leaks), totalling 1804 unique backend storage slots.

The eager path (`runIntegration`) has **zero leaks** — `using` declarations
correctly dispose all temporaries. The reduced scan path (which traces only
the water submodel, not the solver) also has zero leaks.

## Root cause

During tracing:

1. `np.zeros([4, 2])` (or any array-producing call) allocates a real backend
   array.
2. The tracing infrastructure wraps it in a tracer and records it as a constant
   in the jaxpr.
3. User code may apply `using` to the tracer, but `tracer[Symbol.dispose]()`
   is a no-op — it does not touch the underlying constant array.
4. After `lax.scan` / `lax.foriLoop` returns, the compiled function is cached
   with references to the constant arrays. The original concrete arrays are
   never explicitly disposed.

In Python JAX, constants are managed by the XLA runtime and freed when the
compiled function cache is evicted. In jax-js-nonconsuming, no analogous
lifecycle management exists for jaxpr constants.

## Reproduction

```ts
import { checkLeaks, lax, numpy as np } from "@hamk-uas/jax-js-nonconsuming";

checkLeaks.start();

const body = (_i: np.Array, x: np.Array): np.Array => {
  const bias = np.zeros([2]);  // ← becomes a jaxpr constant, never freed
  return x.add(bias);
};

const init = np.zeros([2]);
const result = lax.foriLoop(0, 4, body, init);
result.dispose();
init.dispose();

const report = checkLeaks.stop();
console.log(report.leaked);     // > 0 (the bias constant leaks)
console.log(report.userLeaked);  // > 0 (allocation site is in user code)
```

## Affected user-code sites (full scan integration)

| Source file              | Count | Allocation       |
| ------------------------ | ----: | ---------------- |
| solver.ts (init carry)   |  240  | `np.zeros`       |
| precision.ts             |  103  | `np.array`       |
| aerodynamics.ts          |   24  | `.mul`           |
| solver.ts (body ops)     |   40  | `.astype`, `.sum`|
| integration/yasso/soil   |   10  | misc ops         |

## Desired fix

The library should track jaxpr constants and dispose them when:

- The jaxpr's compiled function is evicted from the cache, **or**
- An explicit cache-clear API is called (e.g., `jit.clearCache()`), **or**
- `checkLeaks.stop()` is called (library could auto-clear for testing)

Alternatively, the library could ref-count constants so that when the last
jaxpr referencing a constant is evicted, the constant is disposed.

## Resolution

**Largely fixed across v0.12.6 and v0.12.7:**

- **v0.12.6** fixed the jit cache const ownership leak via `retainClosedJaxpr()`,
  reducing the 1-day scan integration leak from 1804 slots to 93 slots.
- **v0.12.7** fixed the `UseAfterFreeError` ("Referenced tracer Array has been
  disposed") caused by identity-returning primitives (`stopGradient`,
  `convert_element_type` type match) sharing Array objects across multiple Vars.
  `evalJaxpr` now tracks `valueBindCount` and only disposes when all Var bindings
  are exhausted.

After v0.12.7, all scan-* diagnostic tests pass with zero leaks. The full
1-day scan integration test passes correctness checks; 62 PE intermediate
slots still leak from nested grad/hessian inside traced bodies
(`disposePeIntermediates` skips cleanup inside abstract traces). This is
acknowledged as a known limitation in the test.

That remaining path was rechecked on later releases.

- On **v0.12.8**, the downstream situation did not materially change: the full
   scan integration parity test still reported the known `62`-slot limitation,
   and the targeted solver-in-scan diagnostic was still not leak-clean.
- On **v0.12.10**, the targeted solver-in-scan diagnostic now passes cleanly,
   and the full scan integration parity warning improved from `62` leaked slots
   to `44` leaked slots, but did not reach zero.

So the original traced-jaxpr constant issue remains effectively resolved, but a
smaller remaining traced-control-flow / nested-autodiff leak path is still
tracked separately in issue 004.

As of current `diffSVMC` HEAD on 2026-04-03, the historical downstream
reproductions described in issue 004 no longer reproduce after a local
integration refactor. So there is no currently active downstream blocker to
escalate to `jax-js-nonconsuming` from this repository, even though issue 004
still documents the earlier failure mode and investigation history.

## Workaround

The diagnostic scan reproducer tests now pass on v0.12.7. The remaining
workarounds are:

- the full scan integration parity test, which temporarily suspends leak
   assertions and logs the remaining PE-intermediate leaks while still checking
   eager-vs-scan correctness;
- one targeted solver diagnostic in `phydro-leaf.test.ts` that is still skipped
   when `optimiseMidtermMulti(...)` is exercised inside `lax.scan`.

The remaining PE-intermediate leak path is tracked separately in issue 004,
which now carries the current `v0.12.10` downstream status.
