/**
 * Taylor-based matrix exponential for Yasso20.
 *
 * Fortran reference: vendor/SVMC/src/yassofortran20.f90 L163–202
 *
 * Algorithm: 10-term Taylor series with scaling and squaring.
 *   1. Frobenius norm: p = sqrt(sum(A²))
 *   2. Find scaling exponent j: normiter = 2^j >= p, with j >= 1.
 *   3. Scale: C = A / normiter
 *   4. Taylor: B = I + C + C²/2! + ... + C¹⁰/10!
 *   5. Square: B = B^(2^j)
 *
 * Downstream-owned per the jax-js-nonconsuming PLAN.md.
 * Uses lax.foriLoop with fixed bounds for JIT traceability.
 */

import { DType, lax } from "@hamk-uas/jax-js-nonconsuming";
import { np } from "../precision.js";

/**
 * Maximum squaring iterations for the TypeScript masked-loop fallback.
 *
 * This is an explicit bounded-policy deviation from the exact Fortran loop:
 * `lax.foriLoop` currently needs a static bound here, so we mask extra
 * iterations rather than tracing a data-dependent loop bound. Tests assert
 * that the current reference fixtures stay within this envelope.
 */
const MAX_J = 20;

/** Number of Taylor series terms. */
const Q = 10;

/**
 * Frobenius norm of a square matrix.
 *
 * Fortran: matrixnorm in yassofortran20.f90 L205–215
 */
export function matrixNorm(a: np.Array): np.Array {
  using sq = np.power(a, 2);
  using s = np.sum(sq);
  return np.sqrt(s);
}

/**
 * Compute scaling exponent j such that 2^j >= p, with j >= 1.
 *
 * Analytical equivalent of the Fortran while-loop:
 *   normiter=2, j=1; while p >= normiter: normiter*=2, j+=1
 *
 * Result: j = max(1, floor(log2(max(p, 1))) + 1), as int32.
 */
function scalingExponent(p: np.Array): np.Array {
  using one = np.array(1);
  using pClamped = np.maximum(p, one);
  using log2P = np.log2(pClamped);
  using floored = np.floor(log2P);
  using plusOne = floored.add(1);
  using jFloat = np.maximum(plusOne, one);
  return jFloat.astype(DType.Int32);
}

type TaylorCarry = { b: np.Array; d: np.Array };

type SquareCarry = { b: np.Array };

/**
 * Approximate matrix exponential via Taylor scaling-and-squaring.
 *
 * Matches the Fortran Yasso20 implementation exactly:
 * 10 Taylor terms, Frobenius-norm-based scaling with doubling.
 *
 * Fortran: matrixexp in yassofortran20.f90 L163–202
 *
 * @param a - Square matrix (n×n).
 * @returns Approximate exp(a), same shape as input.
 */
export function matrixExp(a: np.Array): np.Array {
  const n = a.shape[0];

  using p = matrixNorm(a);
  using j = scalingExponent(p);

  // normiter = 2^j (as float for division)
  using two = np.array(2);
  using jFloat = j.astype(a.dtype);
  using normiter = np.power(two, jFloat);

  // Scale: C = A / normiter
  using c = a.div(normiter);

  // Taylor: B = I + C
  using identity = np.eye(n, { dtype: a.dtype });
  using bInit = identity.add(c);

  // Taylor accumulation: add C^k/k! for k=2..Q
  const taylorBody = (i: np.Array, carry: TaylorCarry): TaylorCarry => {
    using iFloat = i.astype(a.dtype);
    using matProd = np.matmul(c, carry.d);
    const newD = matProd.div(iFloat);
    const newB = carry.b.add(newD);
    return { b: newB, d: newD };
  };

  const taylorResult = lax.foriLoop(2, Q + 1, taylorBody, { b: bInit, d: c });
  taylorResult.d.dispose();

  // Squaring: B = B^(2^j), using fixed MAX_J iterations with conditional no-op.
  // Keep MAX_J and the fixture-bound test in sync if this policy changes.
  const squareBody = (i: np.Array, carry: SquareCarry): SquareCarry => {
    using shouldSquare = i.less(j);
    using squared = np.matmul(carry.b, carry.b);
    const newB = np.where(shouldSquare, squared, carry.b);
    return { b: newB };
  };

  const squareResult = lax.foriLoop(0, MAX_J, squareBody, { b: taylorResult.b });
  taylorResult.b.dispose();

  return squareResult.b;
}
