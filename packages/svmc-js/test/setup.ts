import {
  checkLeaks,
  devices,
  getBackend,
} from "@hamk-uas/jax-js-nonconsuming";
import { afterAll, afterEach, beforeEach, expect } from "vitest";
import { np } from "../src/precision.js";

beforeEach(() => {
  checkLeaks.start();
});

afterEach(() => {
  const result = checkLeaks.stop();
  expect(result.leaked, result.summary).toBe(0);
});

// Tear down background workers so vitest/Playwright can exit cleanly.
afterAll(() => {
  for (const dev of devices) {
    try {
      const b = getBackend(dev) as any;
      if (typeof b.destroyWorkers === "function") {
        b.destroyWorkers();
      }
    } catch {
      // Backend may not be initialized; ignore.
    }
  }
});

expect.extend({
  toBeAllclose(
    actual: np.ArrayLike,
    expected: np.ArrayLike,
    options: { rtol?: number; atol?: number; equalNaN?: boolean } = {},
  ) {
    const { isNot } = this;
    const pass = np.allclose(actual, expected, options);
    const actualJs =
      actual != null && typeof (actual as np.Array).js === "function"
        ? (actual as np.Array).js()
        : actual;
    const expectedJs =
      expected != null && typeof (expected as np.Array).js === "function"
        ? (expected as np.Array).js()
        : expected;
    return {
      pass,
      message: () => `expected array to be${isNot ? " not" : ""} allclose`,
      actual: actualJs,
      expected: expectedJs,
    };
  },
});
