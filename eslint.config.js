import { createRequire } from "node:module";
import { join } from "node:path";
import tseslint from "typescript-eslint";

// Resolve from svmc-js where the package is installed
const require = createRequire(
  join(process.cwd(), "packages/svmc-js/package.json"),
);
const jaxJs = require("@hamk-uas/jax-js-nonconsuming/eslint-plugin");

export default [
  // TypeScript parsing for svmc-js
  ...tseslint.configs.recommended.map((c) => ({
    ...c,
    files: ["packages/svmc-js/src/**/*.ts", "packages/svmc-js/test/**/*.ts"],
  })),
  // jax-js ownership rules — recommended catches use-after-dispose (critical),
  // upgrade to jaxJs.strict once no-array-chain / no-nested-array-leak are resolved
  {
    ...jaxJs.recommended,
    files: ["packages/svmc-js/src/**/*.ts", "packages/svmc-js/test/**/*.ts"],
  },
];
