import { describe, expect, it } from "vitest";
import qvidjaRef from "../../../website/public/qvidja-v1-reference.json";
import {
  buildWebsiteJaxInputLike,
  runWebsiteJaxCore,
} from "../../../website/src/qvidja-jax-core.js";

describe("website JAX smoke", () => {
  it("runs the website JAX payload/result path browserlessly for 2 days", async () => {
    const inputLike = buildWebsiteJaxInputLike(qvidjaRef, 2);

    const result = await runWebsiteJaxCore(inputLike, { useJit: true });

    expect(result.executionLabel).toBe("jit");
    expect(result.daily.gpp).toHaveLength(2);
    expect(result.daily.nee).toHaveLength(2);
    expect(result.daily.socTotal).toHaveLength(2);
    expect(Number.isFinite(result.daily.gpp[0])).toBe(true);
    expect(Number.isFinite(result.daily.nee[0])).toBe(true);
    expect(Number.isFinite(result.daily.socTotal.at(-1) ?? NaN)).toBe(true);
  });
});