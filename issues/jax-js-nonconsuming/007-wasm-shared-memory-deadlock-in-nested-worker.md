# ~~WASM backend deadlocks in nested Web Worker when SharedArrayBuffer is available~~ RESOLVED

## Resolution

This was fixed upstream in `jax-js-nonconsuming` at the transport level.
Nested browser workers keep using SharedArrayBuffer-backed WASM memory;
the deadlock was removed by changing orchestrator mega-module registration
to an asynchronous pre-registration step instead of doing a synchronous
`postMessage(module)` followed by a blocking wait.

What changed upstream:

- Parent-side worker/orchestrator waits use `Atomics.wait` when the
  current thread supports it, rather than pure busy polling.
- Orchestrator mega-module registration is now asynchronous.
  The first mega-module call may execute directly while registration is
  in flight; later calls use the synchronous orchestrator dispatch path
  once the module is already resident on the orchestrator worker.
- Shared-memory WASM remains enabled inside nested browser workers when
  SharedArrayBuffer-backed memory is constructible.

What changed downstream:

- Removed the `delete globalThis.SharedArrayBuffer` workaround from
  `website/src/qvidja-jax-worker.js`.

Status:

- Upstream source fix applied.
- Downstream workaround removed.
- Existing downstream smoke path still passes against the linked local
  `../jax-js-nonconsuming` checkout.

## Original Report

## Component

`@hamk-uas/jax-js-nonconsuming` — `WasmBackend` constructor,
`OrchestratorWorker`, `WasmWorkerPool` in `dist/backend-DxA1-Wv9.js`

## Problem

When the WASM backend is instantiated inside a Web Worker on a
CrossOriginIsolated page (localhost, or any page with COOP/COEP
headers), `SharedArrayBuffer` is available.  The backend detects this
via:

```js
capabilities = {
  sharedMemory: (() => {
    try { return new SharedArrayBuffer(1).byteLength === 1; }
    catch { return false; }
  })(),
};
```

and the constructor creates `WebAssembly.Memory({ shared: true })`,
which enables the `OrchestratorWorker` and `WasmWorkerPool` code
paths.

These paths use a spin-wait synchronisation protocol:

1. **`registerModuleSync()`** posts a message to the orchestrator
   worker, then enters a tight `while (Atomics.load(...) === IDLE)`
   loop waiting for a response.
2. **`dispatch()`** posts a message to the orchestrator worker, then
   enters a tight `Atomics.load` loop that also services proxy
   `alloc`/`free` requests via a shared control buffer.

When the caller is itself a Web Worker (i.e. the backend lives in a
_nested_ worker context), the spin-wait blocks the worker's event loop.
The orchestrator worker's `postMessage` responses can never be
delivered because the parent worker is busy-waiting, causing a
permanent deadlock.

### Observed behaviour

```
[worker]  init backend: wasm
[worker]  sharedMemory: true
[worker]  JIT tracing completes normally (all makeJaxpr debug messages appear)
[worker]  bind(Primitive.Jit, ...) — calls jitCompile() → executeMegaModule()
          ... silent for 30 s ...
[worker]  TIMEOUT — runner() never returns
```

JIT tracing (`makeJaxpr`) succeeds because it does not invoke any WASM
kernels.  The hang occurs at the first real mega-module dispatch during
`jp.execute()`.

Eager (non-JIT) execution also deadlocks at the very first kernel
dispatch, confirming the issue is in the WASM backend dispatch path
and not JIT-specific.

The identical workload completes in ~1 s in Node.js (where
`SharedArrayBuffer` is available but there is no nested-worker
constraint).

### Reproduction

Any consumer that runs WASM-backend code inside a Web Worker on a
CrossOriginIsolated page will hit this.  Minimal sketch:

```js
// main.js  (served with COOP/COEP headers, or on localhost)
const worker = new Worker("my-worker.js", { type: "module" });
worker.postMessage({ type: "run" });

// my-worker.js
import { init, np, lax } from "@hamk-uas/jax-js-nonconsuming";
const backend = await init("wasm");
// At this point WasmBackend has sharedMemory: true,
// OrchestratorWorker is active.

const x = np.ones([4]);
const y = np.add(x, x);   // ← deadlocks here (dispatch spin-wait)
```

## Current workaround

We delete `SharedArrayBuffer` from the worker's global scope **before**
calling `init()`, forcing the WASM backend to create non-shared
`WebAssembly.Memory` and skip the orchestrator/worker-pool paths:

```js
if (typeof SharedArrayBuffer !== "undefined") {
  delete globalThis.SharedArrayBuffer;
}
const backend = await init("wasm");
// sharedMemory capability resolves to false → non-shared memory path
```

This is fragile:

- `delete` on a built-in global may not work in all engines.
- It disables SharedArrayBuffer for the entire worker, including any
  legitimate uses.
- Consumer code should not need to know about backend-internal
  threading details.

This is included only to show that the failure is in the backend's
shared-memory/orchestrator path. It is **not** the requested upstream
solution.

## Requested upstream fix

The backend should fix the deadlocking shared-memory execution path,
or disable that path internally when it cannot be used safely. The
problem is that `sharedMemory: true` currently selects a transport that
busy-waits on message delivery from another worker. In a nested worker,
that transport is fundamentally unsafe because the waiting worker blocks
its own ability to receive the response.

What would actually resolve the bug:

1. **Replace the spin-wait message protocol** in
  `registerModuleSync()` / `dispatch()` with a blocking primitive that
  does not starve message delivery, or with a fully async handshake.
  `Atomics.wait` / `Atomics.waitAsync` would be the obvious direction
  if the current design wants to keep shared control buffers.

2. **If nested-worker support is intentionally out of scope**, detect
  that case inside the backend and automatically fall back to the
  non-shared execution path. That fallback should be owned by the
  library, not by consumer code deleting globals.

3. **Report capabilities accurately**. If the backend cannot safely use
  shared-memory orchestration in the current runtime, it should not
  advertise or act on `sharedMemory: true` for that execution mode.

## Non-fixes / temporary mitigations

These may help users unblock themselves, but they should not be treated
as resolution of the bug:

1. **Consumer opt-out flag** such as `init("wasm", { sharedMemory:
  false })`. This is a reasonable escape hatch, but it still pushes
  backend runtime selection onto application code.

2. **Consumer-side environment hacks** like `delete
  globalThis.SharedArrayBuffer` before `init()`. This is the current
  workaround, but it is brittle and masks the backend defect instead of
  fixing it.

## Environment

- Browser: Chromium 136 (headless, via Playwright)
- Page: CrossOriginIsolated (`localhost:5176`, served with
  `Cross-Origin-Opener-Policy: same-origin`,
  `Cross-Origin-Embedder-Policy: require-corp`)
- Backend: WASM (both JIT and eager paths affected)
- Library version: `@hamk-uas/jax-js-nonconsuming@0.1.0-alpha.33`
