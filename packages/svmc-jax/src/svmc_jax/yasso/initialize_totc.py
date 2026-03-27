"""Yasso20 C/N pool initialization from total carbon (initialize_totc).

Fortran reference: vendor/SVMC/src/yasso.f90 L184–236

Initialises the 5-pool AWENH carbon state and the single nitrogen pool
given a prescribed total carbon.  The output is a weighted blend of:
  - an equilibrium partitioning (from a steady-state solve of A·x = -input)
  - a "legacy" partitioning (all carbon in the H pool)

Dependencies:
  - evaluate_matrix_mean_tempr: builds the 5×5 AWENH matrix using 4-point
    annual-cycle temperature averaging (vendor/SVMC/src/yasso.f90 L401–464).
  - eval_steadystate_nitr: iterative CUE-based steady-state N solver
    (vendor/SVMC/src/yasso.f90 L151–182).
  - jnp.linalg.solve: replaces Fortran pgauss/solve for linear system.
"""

import jax
import jax.numpy as jnp

_DAYS_YR = 365.25
_NC_MB = 0.1      # microbial biomass N:C ratio
_CUE_MIN = 0.1    # minimum carbon use efficiency
_NC_H_MAX = 0.1   # maximum N:C ratio of humus pool

# AWENH fractions for input types — currently identical in upstream Fortran.
_AWENH_FINEROOT = jnp.array([0.46, 0.32, 0.04, 0.18, 0.0])
_AWENH_LEAF = jnp.array([0.46, 0.32, 0.04, 0.18, 0.0])

# Legacy state: all carbon in the H pool.
_LEGACY_STATE = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])

_MAX_CUE_ITER = 10
_PI = 3.141592653589793


def _evaluate_matrix_mean_tempr(
    param: jnp.ndarray,
    tempr: jnp.ndarray,
    precip: jnp.ndarray,
    tempr_ampl: jnp.ndarray,
) -> jnp.ndarray:
    """Build the 5×5 AWENH decomposition matrix using 4-point temperature averaging.

    Fortran reference: vendor/SVMC/src/yasso.f90 L401–464

    Unlike the daily-step version (_evaluate_matrix in decompose.py), this uses
    4 strategically placed temperature points to approximate the annual cycle
    instead of a single daily temperature.

    Args:
        param: Parameter vector (35,).
        tempr: Mean annual temperature (°C), scalar.
        precip: Annual precipitation (mm/yr), scalar.
        tempr_ampl: Temperature yearly amplitude (°C), scalar.

    Returns:
        5×5 decomposition matrix A.
    """
    sqrt2 = jnp.sqrt(2.0)

    # 4-point annual cycle approximation
    te = jnp.array([
        tempr + 4 * tempr_ampl * (1 / sqrt2 - 1) / _PI,
        tempr - 4 * tempr_ampl / sqrt2 / _PI,
        tempr + 4 * tempr_ampl * (1 - 1 / sqrt2) / _PI,
        tempr + 4 * tempr_ampl / sqrt2 / _PI,
    ])

    # Average temperature modifiers over the 4 points
    temprm = 0.25 * jnp.sum(jnp.exp(param[21] * te + param[22] * te ** 2))
    temprmN = 0.25 * jnp.sum(jnp.exp(param[23] * te + param[24] * te ** 2))
    temprmH = 0.25 * jnp.sum(jnp.exp(param[25] * te + param[26] * te ** 2))

    # Precipitation modifiers (no /12 division, unlike mod5c20 yearly)
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

    abs_diag = jnp.abs(diag_rates)

    # Build 5×5 matrix
    A = jnp.zeros((5, 5), dtype=param.dtype)
    A = A.at[0, 0].set(diag_rates[0])
    A = A.at[1, 1].set(diag_rates[1])
    A = A.at[2, 2].set(diag_rates[2])
    A = A.at[3, 3].set(diag_rates[3])
    A = A.at[4, 4].set(diag_rates[4])

    # Row 0 (A pool receives from W, E, N)
    A = A.at[0, 1].set(param[4] * abs_diag[1])
    A = A.at[0, 2].set(param[5] * abs_diag[2])
    A = A.at[0, 3].set(param[6] * abs_diag[3])

    # Row 1 (W pool receives from A, E, N)
    A = A.at[1, 0].set(param[7] * abs_diag[0])
    A = A.at[1, 2].set(param[8] * abs_diag[2])
    A = A.at[1, 3].set(param[9] * abs_diag[3])

    # Row 2 (E pool receives from A, W, N)
    A = A.at[2, 0].set(param[10] * abs_diag[0])
    A = A.at[2, 1].set(param[11] * abs_diag[1])
    A = A.at[2, 3].set(param[12] * abs_diag[3])

    # Row 3 (N pool receives from A, W, E)
    A = A.at[3, 0].set(param[13] * abs_diag[0])
    A = A.at[3, 1].set(param[14] * abs_diag[1])
    A = A.at[3, 2].set(param[15] * abs_diag[2])

    # Row 4 (H pool receives from all AWEN; no size effect)
    A = A.at[4, 0].set(param[30] * abs_diag[0])
    A = A.at[4, 1].set(param[30] * abs_diag[1])
    A = A.at[4, 2].set(param[30] * abs_diag[2])
    A = A.at[4, 3].set(param[30] * abs_diag[3])

    return A


def _eval_steadystate_nitr(
    cstate: jnp.ndarray,
    resp_yr: jnp.ndarray,
    nitr_input_yr: jnp.ndarray,
    matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate steady-state nitrogen pool via CUE iteration.

    Fortran reference: vendor/SVMC/src/yasso.f90 L151–182

    Iterates 10 times to converge the CUE-based nitrogen balance.
    Uses jax.lax.fori_loop for differentiability with fixed iteration count.

    Args:
        cstate: AWENH carbon state (5,).
        resp_yr: Yearly respiration in steady state (scalar).
        nitr_input_yr: Yearly nitrogen input (scalar).
        matrix: 5×5 decomposition matrix used for steady state.

    Returns:
        nstate: Steady-state nitrogen (scalar).
    """
    decomp_h = matrix[4, 4] * cstate[4]

    def body(_, cue):
        cupt_awen = (resp_yr - decomp_h) / (1.0 - cue)
        nc_awen = (1.0 / cupt_awen) * (
            _NC_MB * cue * cupt_awen - _NC_H_MAX * decomp_h + nitr_input_yr
        )
        nstate = jnp.sum(cstate[:4]) * nc_awen + _NC_H_MAX * cstate[4]
        nc_som = nstate / jnp.sum(cstate)
        # PORT-BRANCH: CUE clamped to [cue_min, 1.0]
        cue_new = jnp.clip(0.43 * (nc_som / _NC_MB) ** 0.6, _CUE_MIN, 1.0)
        return cue_new

    cue_init = jnp.array(0.43, dtype=cstate.dtype)
    cue_final = jax.lax.fori_loop(0, _MAX_CUE_ITER, body, cue_init)

    # Recompute nstate from converged CUE
    cupt_awen = (resp_yr - decomp_h) / (1.0 - cue_final)
    nc_awen = (1.0 / cupt_awen) * (
        _NC_MB * cue_final * cupt_awen - _NC_H_MAX * decomp_h + nitr_input_yr
    )
    nstate = jnp.sum(cstate[:4]) * nc_awen + _NC_H_MAX * cstate[4]
    return nstate


def initialize_totc(
    param: jnp.ndarray,
    totc: jnp.ndarray,
    cn_input: jnp.ndarray,
    fract_root_input: jnp.ndarray,
    fract_legacy_soc: jnp.ndarray,
    tempr_c: jnp.ndarray,
    precip_day: jnp.ndarray,
    tempr_ampl: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialise Yasso C/N pools from total carbon.

    Fortran reference: vendor/SVMC/src/yasso.f90 L184–236

    The output is a weighted blend of:
      - equilibrium: solve A·x = -input for unit input, scale to match totc
      - legacy: all carbon in H pool, nitrogen at nc_h_max

    Note: In the current Fortran parameterisation, awenh_fineroot == awenh_leaf,
    so fract_root_input has no effect.  We port faithfully regardless.

    Args:
        param: Parameter vector (35,).
        totc: Total carbon pool (scalar).
        cn_input: C:N ratio of the steady-state input (scalar).
        fract_root_input: Fraction of input C with fineroot composition [0,1] (scalar).
            Precondition: must be in [0,1]. Replaces Fortran error_stop guard.
        fract_legacy_soc: Legacy carbon fraction [0,1] (scalar).
            Precondition: must be in [0,1]. Replaces Fortran error_stop guard.
        tempr_c: Mean annual temperature (°C) (scalar).
        precip_day: Precipitation (mm/day) (scalar).
        tempr_ampl: Temperature yearly amplitude (°C) (scalar).

    Returns:
        (cstate, nstate): AWENH carbon state (5,) and nitrogen state (scalar).
    """
    # Build matrix from mean temperature
    precip_yr = precip_day * _DAYS_YR
    matrix = _evaluate_matrix_mean_tempr(param, tempr_c, precip_yr, tempr_ampl)

    # Blend unit input composition
    unit_input = fract_root_input * _AWENH_FINEROOT + (1.0 - fract_root_input) * _AWENH_LEAF

    # Solve for equilibrium partitioning: A · tmpstate = -unit_input
    tmpstate = jnp.linalg.solve(matrix, -unit_input)
    eqfac = totc / jnp.sum(tmpstate)
    eqstate = eqfac * tmpstate

    # Equilibrium nitrogen via CUE iteration
    eqnitr = _eval_steadystate_nitr(eqstate, eqfac, eqfac / cn_input, matrix)

    # Blend equilibrium and legacy
    cstate = fract_legacy_soc * _LEGACY_STATE * totc + (1.0 - fract_legacy_soc) * eqstate
    nstate = fract_legacy_soc * totc * _NC_H_MAX + (1.0 - fract_legacy_soc) * eqnitr

    return cstate, nstate
