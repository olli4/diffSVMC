import {
  _setCalibrationState,
  defaultDevice,
  getBackend,
  init,
} from "@diffsvmc/svmc-js";
import { runWebsiteJaxCore } from "./qvidja-jax-core.js";

const initializedJaxDevices = new Set();

function postDebug(message, details, level = "log") {
  self.postMessage({
    type: "debug",
    message,
    details,
    level,
  });
}

function toErrorMessage(error) {
  if (error instanceof Error) return error.message;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

function installWebGpuDebugHooks() {
  if (typeof navigator === "undefined" || !navigator.gpu) return;
  if (navigator.gpu.__svmcDemoDebugWrapped) return;

  const originalRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);

  navigator.gpu.requestAdapter = async (...args) => {
    const startedAt = performance.now();
    postDebug("webgpu requestAdapter start", { argsLength: args.length });

    try {
      const adapter = await originalRequestAdapter(...args);
      postDebug("webgpu requestAdapter resolved", {
        elapsedMs: performance.now() - startedAt,
        hasAdapter: adapter != null,
      });

      if (!adapter || adapter.__svmcDemoDebugWrapped) {
        return adapter;
      }

      const originalRequestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async (...deviceArgs) => {
        const deviceStartedAt = performance.now();
        postDebug("webgpu requestDevice start", {
          argsLength: deviceArgs.length,
        });

        try {
          const device = await originalRequestDevice(...deviceArgs);
          postDebug("webgpu requestDevice resolved", {
            elapsedMs: performance.now() - deviceStartedAt,
            hasDevice: device != null,
          });
          return device;
        } catch (error) {
          postDebug("webgpu requestDevice failed", { message: toErrorMessage(error) }, "error");
          throw error;
        }
      };
      adapter.__svmcDemoDebugWrapped = true;
      return adapter;
    } catch (error) {
      postDebug("webgpu requestAdapter failed", { message: toErrorMessage(error) }, "error");
      throw error;
    }
  };

  navigator.gpu.__svmcDemoDebugWrapped = true;
}

const FALLBACK_ORDER = ["webgpu", "wasm", "cpu"];

async function ensureJaxDeviceReady(device) {
  postDebug("ensureJaxDeviceReady called", {
    device,
    alreadyInitialized: initializedJaxDevices.has(device),
  });
  if (initializedJaxDevices.has(device)) return device;

  const initStartedAt = performance.now();
  const available = await init(device);
  postDebug("init(device) resolved", {
    device,
    elapsedMs: performance.now() - initStartedAt,
    available,
  });
  if (available.includes(device)) {
    initializedJaxDevices.add(device);
    return device;
  }

  // Auto-fallback to next available backend
  for (const fallback of FALLBACK_ORDER) {
    if (fallback !== device && available.includes(fallback)) {
      postDebug("backend fallback", { requested: device, fallback, available });
      initializedJaxDevices.add(fallback);
      return fallback;
    }
  }
  throw new Error(`No backend available (requested ${device}, found ${available.join(", ")}).`);
}

_setCalibrationState("off");
installWebGpuDebugHooks();
globalThis.__SVMC_DEMO_DEBUG__ = (message, details) => {
  postDebug(message, details);
};

postDebug("worker module loaded");
postDebug("webgpu calibration disabled for worker demo");

self.addEventListener("message", async (event) => {
  const message = event.data;
  if (!message || message.type !== "run") return;

  const { device: requestedDevice, useJit, inputLike } = message.payload;

  try {
    postDebug("worker run start", {
      device: requestedDevice,
      useJit,
      ndays: inputLike.daily_lai.length,
    });

    const device = await ensureJaxDeviceReady(requestedDevice);
    defaultDevice(device);
    postDebug("after defaultDevice", { activeDevice: device, requested: requestedDevice });

    const backend = getBackend(device);
    postDebug("after getBackend", {
      device,
      backendType: backend.type,
      capabilities: {
        sharedMemory: backend.capabilities.sharedMemory,
        multiOutputKernel: backend.capabilities.multiOutputKernel,
        shaderF16: backend.capabilities.shaderF16,
        atomicF32Add: backend.capabilities.atomicF32Add,
        calibrated: backend.capabilities.calibrated ?? false,
      },
    });

    const result = await runWebsiteJaxCore(inputLike, {
      useJit,
      logger: (debugMessage, details) => postDebug(debugMessage, details),
    });

    self.postMessage({
      type: "result",
      result: {
        elapsedMs: result.elapsedMs,
        executionLabel: `${backend.type} + ${result.executionLabel}`,
        daily: result.daily,
      },
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: toErrorMessage(error),
    });
  } finally {
    postDebug("worker cleanup done");
  }
});
