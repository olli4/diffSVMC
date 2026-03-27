import {
  Array as NpArray,
  type ArrayLike as NpArrayLike,
  DType,
  numpy as baseNp,
} from "@hamk-uas/jax-js-nonconsuming";

export type NumericDType = DType.Float32 | DType.Float64;

const DEFAULT_NUMERIC_DTYPE: NumericDType = DType.Float32;

type EnvRecord = Record<string, string | undefined>;

const configuredNumericDType = readNumericDTypeFromEnv();

function readNumericDTypeFromEnv(): NumericDType {
  const importMetaEnv = (import.meta as ImportMeta & { env?: EnvRecord }).env;
  const processEnv = (globalThis as typeof globalThis & {
    process?: { env?: EnvRecord };
  }).process?.env;
  const raw = importMetaEnv?.SVMC_JS_DTYPE?.toLowerCase()
    ?? processEnv?.SVMC_JS_DTYPE?.toLowerCase();
  return raw === DType.Float64 ? DType.Float64 : DEFAULT_NUMERIC_DTYPE;
}

function isProjectArray(value: unknown): boolean {
  return value instanceof NpArray;
}

function isBooleanTree(value: unknown): boolean {
  if (typeof value === "boolean") return true;
  if (Array.isArray(value)) return value.length > 0 && value.every(isBooleanTree);
  return false;
}

function shouldApplyNumericDType(value: unknown, dtype: unknown): boolean {
  if (dtype != null) return false;
  if (isProjectArray(value)) return false;
  if (ArrayBuffer.isView(value)) return false;
  if (isBooleanTree(value)) return false;
  return typeof value === "number" || Array.isArray(value);
}

function createArrayForDType(dtype: NumericDType): typeof baseNp.array {
  return (values, options) => {
    if (!shouldApplyNumericDType(values, options?.dtype)) {
      return baseNp.array(values as never, options as never);
    }
    return baseNp.array(values as never, { ...options, dtype } as never);
  };
}

export function createPrecisionNp(dtype: NumericDType): typeof baseNp {
  const array = createArrayForDType(dtype);
  return new Proxy(baseNp as object, {
    get(target, prop, receiver) {
      if (prop === "array") return array;
      return Reflect.get(target, prop, receiver);
    },
  }) as typeof baseNp;
}

export const array = createArrayForDType(configuredNumericDType);

export const np = createPrecisionNp(configuredNumericDType);

export namespace np {
  export type Array = NpArray;
  export type ArrayLike = NpArrayLike;
}

export function getNumericDType(): NumericDType {
  return configuredNumericDType;
}
