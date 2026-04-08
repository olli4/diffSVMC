# Scan Ownership Plan for diffSVMC

**Goal:** One implementation of the integration loop. Eager and jitted execution use the same code.
The only difference is whether an outer `jit()` wraps the call.

## Status (2026-04-03)

### What works

- `using` is safe for tracer values (`Tracer.[Symbol.dispose]()` is a no-op).
- `optimiseMidtermMulti(...)` works inside `lax.scan` when parameter objects are built from arrays
  outside the traced body.
- The full 1-day `runIntegrationScanExperimental(...)` path passes. The parity test against eager
  replay passes within 6%/1e-12 tolerance.
- `grad()`/`hessian()` inside traced scan/foriLoop bodies work after the jax-js v0.12.8 PE cleanup
  fix (`disposePeIntermediates` narrowed to allow nested autodiff in traced control-flow bodies).
- Use `asArray(...)` from `src/precision.ts` when code may receive either a raw scalar/JS array or
  an existing jax array.

### What remains

- `solver.ts` still has module-level `jit()` wrapping (`lbfgsSolve`, `projectedNewtonSolve`).
- The eager path (`runIntegration`) and scan path (`runIntegrationScanExperimental`) are separate
  ~300 LOC functions with duplicated hourly/daily logic.
- The eager path uses JS primitives (numbers) for carry state, wrapping/unwrapping `np.array()` per
  iteration. The scan path uses `np.Array` throughout.

---

## Current architecture (two paths)

```
runIntegration (eager)                   runIntegrationScanExperimental (scan)
│                                        │
├─ for day                               ├─ lax.scan(dailyStep, init, forcing)
│  ├─ for hour                           │  └─ dailyStep(carry, forcing)
│  │  ├─ wrap JS → np.array              │     ├─ lax.foriLoop(0, 24, hourlyBody, ...)
│  │  ├─ pmodelHydraulics(...)           │     │  └─ pmodelHydraulics(...)
│  │  ├─ canopyWaterFlux(...)            │     │     canopyWaterFlux(...)
│  │  ├─ soilWater(...)                  │     │     soilWater(...)
│  │  ├─ unwrap np.array → JS            │     ├─ invertAllocFn(...)
│  │  └─ .dispose() all intermediates    │     ├─ allocHypothesis2Fn(...)
│  ├─ invertAllocFn(...)                 │     ├─ decomposeFn(...)
│  ├─ allocHypothesis2Fn(...)            │     └─ return [next_carry, output]
│  ├─ decomposeFn(...)                   │
│  └─ .dispose() all intermediates       └─ return [finalCarry, outputs]
└─ return results
```

The kernel functions (`pmodelHydraulicsNumerical`, `canopyWaterFlux`, `soilWater`,
`invertAllocFn`, `allocHypothesis2Fn`, `decomposeFn`) are shared. The duplication is in the loop
structure, carry management, and ownership plumbing.

---

## Target architecture (one path)

```
runIntegration(inputs, { jit: false })          ← eager: runs dailyStep directly
runIntegration(inputs, { jit: true  })          ← scan:  jit(lax.scan(dailyStep, ...))

Both call the SAME dailyStep(carry, forcing) function.
```

### The key insight

Write `dailyStep` and `hourlyBody` using the **array API only** (no raw JS scalars, no manual
disposal, no `.ref`). This code is ownership-correct in both modes:

- **Eager:** `using` cleans up intermediates. Carry arrays survive because they are returned (not
  `using`'d). Caller disposes final outputs.
- **JIT/scan:** `using` is a no-op on tracers. The compiler manages all lifetimes. `lax.scan`
  handles carry transfer and Y stacking.

The **only** difference between the two modes is the outer call site:

```ts
// Eager: plain JS loop
function runIntegrationEager(inputs) {
  const init = buildDailyCarry(inputs);
  let carry = init;
  const outputs = [];
  for (let d = 0; d < ndays; d++) {
    const [newCarry, output] = dailyStep(carry, forcing[d]);
    carry.dispose();          // ← only in eager
    carry = newCarry;
    outputs.push(output);
  }
  return [carry, outputs];
}

// Scan: lax.scan compiles the loop
function runIntegrationScan(inputs) {
  const init = buildDailyCarry(inputs);
  return lax.scan(dailyStep, init, dailyForcing, { length: ndays });
}
```

Both call `dailyStep`. The function itself doesn't know or care which mode it's in.

### What `dailyStep` looks like in the converged design

```ts
function dailyStep(carry: DailyCarry, forcing: DailyForcing): [DailyCarry, DailyOutput] {
  // Compute daily parameters from carry + forcing
  using fapar = computeFapar(carry.lai, ...);
  using delta_lai = computeDeltaLai(...);

  // Hourly loop — same function in both modes
  const hourlyInit = buildHourlyCarry(carry, forcing);
  const [finalHourly, _] = lax.foriLoop(0, 24, hourlyBody, hourlyInit);

  // Daily post-processing
  using allocResult = allocHypothesis2Fn(...);
  using decompResult = decomposeFn(...);
  // ...

  const nextCarry: DailyCarry = { /* fields from finalHourly, allocResult, ... */ };
  const output: DailyOutput = { /* fields for this day */ };

  return [nextCarry, output];
}
```

Rules inside this function:

| Pattern | Status | Why |
|---------|--------|-----|
| `using temp = someOp(...)` | OK | No-op on tracers; cleans up in eager |
| `const result = someOp(...)` | OK | Caller/scan manages lifetime |
| `.dispose()` on intermediates | REMOVE | Kills tracers under scan |
| `.ref` on carry/output fields | REMOVE | Non-consuming API; not needed |
| `np.array(existingArray)` | AVOID | Returns same object; disposing "copy" disposes original |
| `jit()` inside body | REMOVE | Outer scan already compiles everything |
| `asArray(value)` | PREFER | Safe for both raw scalars and existing arrays |
| `lax.foriLoop(...)` inside body | OK | Nested loops compose correctly |
| `grad()`/`hessian()` inside body | OK | Works after jax-js v0.12.8 PE fix |

---

## Migration plan

### Phase 1: Unify the body functions

Extract shared `dailyStep(carry, forcing)` and `hourlyBody(i, carry)` that work in both modes.

1. **Convert the eager path to use `np.Array` carry** instead of JS primitives. The eager path
   currently wraps numbers into `np.array()` each iteration and unwraps back. Change it to keep
   arrays throughout, like the scan path.

2. **Remove `.dispose()` / `.ref` from body functions.** Use `using` for temporaries. Don't dispose
   carry or output fields — the caller handles that.

3. **Remove `jit()` from solver.ts.** The module-level `jit(lbfgsSolveImpl)` and
   `jit(projectedNewtonSolveImpl)` must become plain function calls. When called from inside
   `lax.scan`, the outer trace compiles everything — nested `jit()` creates tracing-level conflicts.

4. **Validate:** `dailyStep` runs correctly when called both directly (eager loop) and via
   `lax.scan`.

### Phase 2: Unify the entry point

1. **Write a single `runIntegration(inputs, opts)`** that either:
   - Loops in JS calling `dailyStep` directly (eager), or
   - Calls `lax.scan(dailyStep, init, forcing)` (scan/jit)

   The `opts.jit` flag (or similar) selects the mode. Both share `buildDailyCarry(inputs)` for
   initialization.

2. **Delete `runIntegrationScanExperimental`.** It becomes the `jit: true` branch of
   `runIntegration`.

3. **Update tests.** The parity test becomes: run `runIntegration` with `jit: false` and
   `jit: true`, compare outputs. Same function, same body, different loop driver.

### Phase 3: Ownership cleanup

1. **Eager loop caller manages carry disposal.** After each iteration, dispose the old carry before
   replacing it with the new one. This is the caller's job, not `dailyStep`'s.

2. **Remove `cloneDailyCarry` / `cloneDailyOutput` helpers.** Not needed — the scan compiler
   manages carry transfer, and the eager loop just moves ownership.

3. **Run `checkLeaks` validation.** Both modes should produce identical leak counts ($\pm$ 0).

---

## What changes and what stays

| Component | Before | After |
|-----------|--------|-------|
| `dailyStep` | Two copies (eager inline + scan closure) | One shared function |
| `hourlyBody` | Two copies | One shared function |
| Carry state type | JS primitives (eager) vs `np.Array` (scan) | `np.Array` in both |
| Ownership in body | `.dispose()`/`.ref` (eager), implicit (scan) | `using` only, no `.dispose()`/`.ref` |
| `solver.ts` | `jit(lbfgsSolveImpl)` | `lbfgsSolveImpl` (plain) |
| Entry points | `runIntegration` + `runIntegrationScanExperimental` | `runIntegration(inputs, { jit })` |
| Kernel functions | Shared | Shared (unchanged) |
| Tests | Separate eager/scan tests | One test, two modes |

## What does NOT change

- The kernel functions (`pmodelHydraulicsNumerical`, `canopyWaterFlux`, `soilWater`,
  `invertAllocFn`, `allocHypothesis2Fn`, `decomposeFn`) — already shared.
- Parameter setup and initialization — stays outside the loop body.
- The `finally` block that disposes `soil_params`, `aero_params`, etc. — stays.
- `lax.scan` / `lax.foriLoop` semantics in jax-js — no upstream changes needed.

---

## Constraints

1. **`using` is safe in both modes.** Tracer disposal is a no-op; in eager mode it cleans up
   correctly. This is the primary ownership mechanism inside body functions.

2. **No `.dispose()` on values that might be carry or output.** The caller (JS loop or `lax.scan`)
   manages those lifetimes. Only use `using` for local temporaries that are consumed within the
   same scope.

3. **`lax.foriLoop` for the hourly loop, not a JS `for`.** Even in "eager" mode, using
   `lax.foriLoop` is fine — it runs the body directly when not under a trace. This avoids a second
   divergence point. If eager perf without JIT matters, a JS `for` loop calling the same
   `hourlyBody` is also fine since the function is mode-agnostic.

4. **`asArray()` for values that might be scalars or arrays.** Avoids the `np.array(existingArray)`
   identity trap.

5. **Solver `jit()` removal is a prerequisite.** Nothing else converges until the nested
   `jit()` is gone.
