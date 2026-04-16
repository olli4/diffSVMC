"""Yasso20 yearly ODE solver (mod5c20).

Fortran reference: vendor/SVMC/src/yassofortran20.f90 L24–161

Solves the linear ODE: x'(t) = A(θ) · x(t) + b, x(0) = init
  - Transient: x(t) = A⁻¹ · (exp(At) · (A·init + b) − b)
  - Steady state: x∞ = −A⁻¹ · b

The coefficient matrix A is built from Yasso20 parameters θ, monthly
temperatures, annual precipitation, size class, and leaching rate.
"""

import jax.numpy as jnp

from ._matrix import build_awenh_matrix
from .matrixexp import matrixexp, matrixnorm


_TOL = 1e-12

# Threshold for switching between Taylor and direct computation of
# (exp(At) - I)·b.  Below this, the subtraction exp(At)·b - b cancels
# catastrophically; above it, the direct subtraction is accurate.
# sqrt(eps) balances Taylor truncation error ≈ ||At||² against
# cancellation error ≈ eps / ||At||.
_NORM_SWITCH = jnp.sqrt(jnp.finfo(jnp.float64).eps)


def _build_coefficient_matrix(
    theta: jnp.ndarray,
    temp: jnp.ndarray,
    prec: jnp.ndarray,
    d: jnp.ndarray,
    leac: jnp.ndarray,
) -> jnp.ndarray:
    """Build the 5×5 AWENH decomposition matrix A.

    Fortran: mod5c20 matrix-construction block, L70–145
    """
    # Temperature dependence: average over 12 months
    tem = jnp.sum(jnp.exp(theta[21] * temp + theta[22] * temp ** 2))
    temN = jnp.sum(jnp.exp(theta[23] * temp + theta[24] * temp ** 2))
    temH = jnp.sum(jnp.exp(theta[25] * temp + theta[26] * temp ** 2))

    # Precipitation dependence
    tem = tem * (1.0 - jnp.exp(theta[27] * prec / 1000.0)) / 12.0
    temN = temN * (1.0 - jnp.exp(theta[28] * prec / 1000.0)) / 12.0
    temH = temH * (1.0 - jnp.exp(theta[29] * prec / 1000.0)) / 12.0

    # Size class dependence
    size_dep = jnp.minimum(
        1.0,
        (1.0 + theta[32] * d + theta[33] * d ** 2) ** (-jnp.abs(theta[34])),
    )

    # Diagonal: decomposition rates (AWEN)
    alpha = jnp.abs(theta[:4])
    diag_rates = jnp.array([
        -alpha[0] * tem * size_dep,
        -alpha[1] * tem * size_dep,
        -alpha[2] * tem * size_dep,
        -alpha[3] * temN * size_dep,
        -jnp.abs(theta[31]) * temH,
    ])

    # Leaching (AWEN only, not H)
    leac_term = leac * prec / 1000.0
    return build_awenh_matrix(
        diag_rates,
        theta[4:16],
        theta[30],
        leac_term=leac_term,
    )


def mod5c20(
    theta: jnp.ndarray,
    time: jnp.ndarray,
    temp: jnp.ndarray,
    prec: jnp.ndarray,
    init: jnp.ndarray,
    b: jnp.ndarray,
    d: jnp.ndarray,
    leac: jnp.ndarray,
    steadystate_pred: bool = False,
) -> jnp.ndarray:
    """Yasso20 yearly ODE solver.

    Fortran: mod5c20 in yassofortran20.f90 L24–161

    Args:
        theta: Parameter vector (35,).
        time: Integration time (years).
        temp: Monthly mean temperatures (12,).
        prec: Annual precipitation (mm), scalar.
        init: Initial AWENH state (5,).
        b: Annual C input (5,).
        d: Decomposing organic matter size (g cm⁻³), scalar.
        leac: Leaching parameter, scalar.
        steadystate_pred: If True, compute steady-state x = −A⁻¹b.

    Returns:
        AWENH state (5,) after integration.
    """
    A = _build_coefficient_matrix(theta, temp, prec, d, leac)

    # Temperature multiplier check: if tem ≤ tol, no decomposition
    tem = jnp.sum(jnp.exp(theta[21] * temp + theta[22] * temp ** 2))
    tem = tem * (1.0 - jnp.exp(theta[27] * prec / 1000.0)) / 12.0

    def _early_return():
        return init + b * time

    def _solve():
        return jnp.where(
            steadystate_pred,
            _solve_steady(A, b),
            _solve_transient(A, time, init, b),
        )

    return jnp.where(tem <= _TOL, _early_return(), _solve())


def _solve_steady(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Steady-state: x = −A⁻¹b, i.e. solve(−A, b)."""
    return jnp.linalg.solve(-A, b)


def _solve_transient(
    A: jnp.ndarray,
    time: jnp.ndarray,
    init: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Transient: x(t) = A⁻¹ · (exp(At)·(A·init + b) − b).

    Split z₂ = exp(At)·z₁ − b into two parts to avoid catastrophic
    cancellation when ||At|| is small (exp(At) ≈ I makes exp(At)·b − b ≈ 0):
      z₂ = exp(At)·(A·init) + (exp(At) − I)·b
                   ↑ stable        ↑ use Taylor when small
    """
    ainit = jnp.matmul(A, init)
    At = A * time
    mexpAt = matrixexp(At)
    z2_init = jnp.matmul(mexpAt, ainit)

    # (exp(At) − I)·b: Taylor gives At·b; direct gives exp(At)·b − b.
    z2_b_direct = jnp.matmul(mexpAt, b) - b
    z2_b_taylor = jnp.matmul(At, b)
    norm_at = matrixnorm(At)
    z2_b = jnp.where(norm_at <= _NORM_SWITCH, z2_b_taylor, z2_b_direct)

    z2 = z2_init + z2_b
    return jnp.linalg.solve(A, z2)
