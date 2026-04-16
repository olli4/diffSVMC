"""Yasso20 daily C/N decomposition step (decompose).

Fortran reference: vendor/SVMC/src/yasso.f90 L255–318

Computes daily carbon and nitrogen tendencies using first-order explicit
Euler integration of the 5×5 AWENH decomposition matrix.  Unlike mod5c20,
this uses a single daily temperature (not monthly averages), has no size
class dependence, and no leaching terms.

Nitrogen dynamics follow CUE-based immobilization/mineralization.
"""

import jax.numpy as jnp

from ._matrix import build_awenh_matrix

_DAYS_YR = 365.25
_NC_MB = 0.1     # microbial biomass N:C ratio
_CUE_MIN = 0.1   # minimum carbon use efficiency
_NC_H_MAX = 0.1  # maximum N:C ratio of humus pool


def _evaluate_matrix(
    param: jnp.ndarray,
    tempr: jnp.ndarray,
    precip: jnp.ndarray,
) -> jnp.ndarray:
    """Build the 5×5 AWENH decomposition matrix A for daily decomposition.

    Fortran reference: vendor/SVMC/src/yasso.f90 L339–391

    Unlike the mod5c20 matrix construction, this version:
      - Uses a single daily temperature (no monthly averaging, no /12)
      - Has no size class dependence
      - Has no leaching terms

    Args:
        param: Parameter vector (35,).
        tempr: Air temperature (°C), scalar.
        precip: Precipitation (mm/yr), scalar.

    Returns:
        5×5 decomposition matrix A.
    """
    # Temperature modifiers
    temprm = jnp.exp(param[21] * tempr + param[22] * tempr ** 2)
    temprmN = jnp.exp(param[23] * tempr + param[24] * tempr ** 2)
    temprmH = jnp.exp(param[25] * tempr + param[26] * tempr ** 2)

    # Precipitation modifiers (no /12 division unlike mod5c20)
    decm = temprm * (1.0 - jnp.exp(param[27] * precip * 0.001))
    decmN = temprmN * (1.0 - jnp.exp(param[28] * precip * 0.001))
    decmH = temprmH * (1.0 - jnp.exp(param[29] * precip * 0.001))

    # Diagonal decomposition rates
    alpha = jnp.abs(param[:4])
    diag_rates = jnp.array([
        -alpha[0] * decm,
        -alpha[1] * decm,
        -alpha[2] * decm,
        -alpha[3] * decmN,
        -jnp.abs(param[31]) * decmH,
    ])

    return build_awenh_matrix(diag_rates, param[4:16], param[30])


def decompose(
    param: jnp.ndarray,
    timestep_days: jnp.ndarray,
    tempr_c: jnp.ndarray,
    precip_day: jnp.ndarray,
    cstate: jnp.ndarray,
    nstate: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Daily C/N decomposition step.

    Fortran reference: vendor/SVMC/src/yasso.f90 L255–318

    Computes carbon tendencies via first-order Euler integration:
      ctend = A · cstate · (timestep_days / 365.25)
    and nitrogen tendency via CUE-based immobilization/mineralization.

    All branches use jnp.where for differentiability.

    Args:
        param: Parameter vector (35,).
        timestep_days: Timestep length (days), scalar.
        tempr_c: Air temperature (°C), scalar.
        precip_day: Precipitation (mm/day), scalar.
        cstate: AWENH carbon state (5,).
        nstate: Nitrogen state (single pool), scalar.

    Returns:
        (ctend, ntend): Carbon tendencies (5,) and nitrogen tendency (scalar).
    """
    totc = jnp.sum(cstate)

    # Carbon: explicit Euler
    precip_yr = precip_day * _DAYS_YR
    matrix = _evaluate_matrix(param, tempr_c, precip_yr)
    timestep_yr = timestep_days / _DAYS_YR
    ctend = jnp.matmul(matrix, cstate) * timestep_yr
    resp = jnp.sum(-ctend)

    # Nitrogen dynamics (branch-free for autodiff)
    # PORT-BRANCH: totc < 1e-6 → ntend = 0 (no SOM, no N dynamics)
    decomp_h = matrix[4, 4] * cstate[4] * timestep_yr

    # PORT-BRANCH: cstate[4]*nc_h_max > nstate → nc_h = nstate/totc
    nc_h_normal = _NC_H_MAX
    nc_h_unusual = nstate / (totc + 1e-30)  # guard division by zero
    nc_h = jnp.where(cstate[4] * _NC_H_MAX > nstate, nc_h_unusual, nc_h_normal)

    nitr_awen = nstate - cstate[4] * nc_h
    nc_awen = nitr_awen / (totc - cstate[4] + 1e-9)
    nc_som = nstate / (totc + 1e-30)  # guard division by zero

    # PORT-BRANCH: CUE clamped to [cue_min, 1.0]
    cue = jnp.clip(0.43 * (nc_som / _NC_MB) ** 0.6, _CUE_MIN, 1.0)

    cupt_awen = (resp - decomp_h) / (1.0 - cue)
    growth_c = cue * cupt_awen
    ntend_computed = _NC_MB * growth_c - nc_awen * cupt_awen - nc_h * decomp_h

    # PORT-BRANCH: totc < 1e-6 → ntend = 0
    ntend = jnp.where(totc < 1e-6, 0.0, ntend_computed)

    return ctend, ntend
