import { defineConfig } from "vitest/config";

export default defineConfig({
  esbuild: {
    supported: {
      using: false, // Lower 'using' to try/finally for correct disposal.
    },
  },
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  test: {
    // Default to Node for non-browser tests (JAX reference comparison etc.)
    // Browser tests for jax-js-nonconsuming packages use workspace-level overrides.
    include: ["packages/*/test/**/*.test.ts"],
    setupFiles: ["packages/svmc-js/test/setup.ts"],
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false, // No visual UI — screenshots are worthless
      name: "chromium",
      provider: "playwright",
      providerOptions: {
        launch: {
          args: [
            "--no-sandbox",
            "--headless=new",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--disable-vulkan-surface",
            "--enable-unsafe-webgpu",
          ],
          env: {
            DISPLAY: process.env.DISPLAY ?? ":0",
            XAUTHORITY:
              process.env.XAUTHORITY ??
              `/run/user/${process.getuid?.() ?? 1000}/gdm/Xauthority`,
          },
        },
      },
    },
  },
});
