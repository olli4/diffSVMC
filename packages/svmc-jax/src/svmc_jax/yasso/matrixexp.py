"""Taylor-based matrix exponential for Yasso20.

Fortran reference: vendor/SVMC/src/yassofortran20.f90 L163–202

Algorithm: 10-term Taylor series with scaling and squaring.
  1. Frobenius norm: p = sqrt(sum(A**2))
  2. Find scaling exponent j: normiter = 2^j >= p, with j >= 1.
  3. Scale: C = A / normiter
  4. Taylor: B = I + C + C^2/2! + ... + C^10/10!
  5. Square: B = B^(2^j)

This implementation is downstream-owned per the jax-js-nonconsuming PLAN.md.
The scaling exponent is computed analytically (no while_loop) so the same
approach works in both JAX and jax-js-nonconsuming's foriLoop.
"""

import jax
import jax.numpy as jnp


def matrixnorm(a: jnp.ndarray) -> jnp.ndarray:
    """Frobenius norm of a square matrix.

    Fortran: matrixnorm in yassofortran20.f90 L205–215
    """
    return jnp.sqrt(jnp.sum(a ** 2))


def _scaling_exponent(p: jnp.ndarray) -> jnp.ndarray:
    """Compute scaling exponent j such that 2^j >= p, with j >= 1.

    Matches the Fortran while-loop:
      normiter=2, j=1; while p >= normiter: normiter*=2, j+=1

    After the loop: normiter = 2^j, p < 2^j.
    Analytically: j = max(1, floor(log2(max(p, 1))) + 1).
    """
    return jnp.maximum(1, jnp.floor(jnp.log2(jnp.maximum(p, 1.0))) + 1).astype(jnp.int32)


def matrixexp(a: jnp.ndarray) -> jnp.ndarray:
    """Approximate matrix exponential via Taylor scaling-and-squaring.

    Matches the Fortran Yasso20 implementation exactly:
    10 Taylor terms, Frobenius-norm-based scaling with doubling.

    Fortran: matrixexp in yassofortran20.f90 L163–202

    Args:
        a: Square matrix (n×n).

    Returns:
        Approximate exp(a), same shape as input.
    """
    n = a.shape[0]
    q = 10  # number of Taylor terms

    p = matrixnorm(a)
    j = _scaling_exponent(p)
    normiter = jnp.float32(2) ** j

    # Scale
    c = a / normiter

    # Taylor accumulation: B = I + C, then add C^k/k! for k=2..q
    identity = jnp.eye(n, dtype=a.dtype)
    b_init = identity + c
    d_init = c

    def _taylor_body(i, carry):
        b, d = carry
        d = jnp.matmul(c, d) / i.astype(a.dtype)
        b = b + d
        return b, d

    b, _ = jax.lax.fori_loop(2, q + 1, _taylor_body, (b_init, d_init))

    # Squaring: B = B^(2^j)
    def _square_body(_, b_):
        return jnp.matmul(b_, b_)

    b = jax.lax.fori_loop(0, j, _square_body, b)

    return b
