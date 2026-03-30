"""P-Hydro solver: differentiable optimization using JAXopt L-BFGS-B.

Replaces Fortran's finite-difference L-BFGS-B with JAXopt's traced
L-BFGS-B solver using autodiff gradients through fn_profit and implicit
differentiation through the optimizer solution.  This keeps the solver
box-constrained like the Fortran reference while preserving end-to-end
differentiability for outer calibration loops.

Port of optimise_midterm_multi and pmodel_hydraulics_numerical from
phydro_mod.f90 in the Fortran SVMC model.
"""

import jax
import jax.numpy as jnp
from jaxopt import LBFGSB

from svmc_jax.phydro.leaf_functions import (
    ParCost,
    ParEnv,
    ParPhotosynth,
    ParPlant,
    calc_assim_light_limited,
    calc_gs,
    calc_kmm,
    density_h2o,
    fn_profit,
    ftemp_kphio,
    gammastar,
    viscosity_h2o,
)


# Fortran defaults from readvegpara_mod
KPHIO = 0.087182

# Fortran box bounds for the optimisation variables
_LOG_JMAX_LO = -10.0
_LOG_JMAX_HI = 10.0
_DPSI_LO = 1.0e-4
_DPSI_HI = 1.0e6

_X0 = jnp.array([4.0, 1.0], dtype=jnp.float64)
_BOUNDS = (
    jnp.array([_LOG_JMAX_LO, _DPSI_LO], dtype=jnp.float64),
    jnp.array([_LOG_JMAX_HI, _DPSI_HI], dtype=jnp.float64),
)


def _objective(
    params: jax.Array,
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> jax.Array:
    """Bound-constrained profit objective for JAXopt L-BFGS-B."""
    return fn_profit(
        params[0],
        params[1],
        psi_soil,
        par_cost,
        par_photosynth,
        par_plant,
        par_env,
        do_optim=True,
    )


_LBFGSB = LBFGSB(
    fun=_objective,
    maxiter=64,
    tol=1.0e-3,
    history_size=10,
    implicit_diff=True,
    jit=True,
    verbose=False,
)


@jax.jit
def optimise_midterm_multi(
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> tuple[jax.Array, jax.Array]:
    """Find optimal (log_jmax, dpsi) that maximise the profit function.

    Uses JAXopt's traced L-BFGS-B solver with native box constraints and
    implicit differentiation through the optimizer solution.  Unlike the
    previous callback-based path, this remains compatible with ``jax.grad``
    through the full solver and integration loop.

    Returns:
        (log_jmax_opt, dpsi_opt) — the optimised parameters
    """
    opt_step = _LBFGSB.run(
        _X0,
        _BOUNDS,
        psi_soil,
        par_cost,
        par_photosynth,
        par_plant,
        par_env,
    )
    return opt_step.params[0], opt_step.params[1]


def pmodel_hydraulics_numerical(
    tc: jax.Array,
    ppfd: jax.Array,
    vpd: jax.Array,
    co2: jax.Array,
    sp: jax.Array,
    fapar: jax.Array,
    psi_soil: jax.Array,
    rdark_leaf: jax.Array,
    conductivity: float = 4e-16,
    psi50: float = -3.46,
    b_param: float = 2.0,
    alpha: float = 0.1,
    gamma_cost: float = 0.5,
    kphio: float = KPHIO,
) -> dict[str, jax.Array]:
    """Full P-Hydro solver: compute optimal photosynthesis-hydraulics state.

    Port of pmodel_hydraulics_numerical from phydro_mod.f90.

    Returns dict with keys: jmax, dpsi, gs, aj, ci, chi, vcmax, profit,
    chi_jmax_lim.
    """
    tc = jnp.float64(tc)
    sp = jnp.float64(sp)
    vpd = jnp.float64(vpd)
    ppfd = jnp.float64(ppfd)
    co2 = jnp.float64(co2)
    fapar = jnp.float64(fapar)
    psi_soil = jnp.float64(psi_soil)
    rdark_leaf = jnp.float64(rdark_leaf)

    # Build parameter structs
    par_plant = ParPlant(
        conductivity=jnp.float64(conductivity),
        psi50=jnp.float64(psi50),
        b=jnp.float64(b_param),
    )
    par_cost = ParCost(
        alpha=jnp.float64(alpha),
        gamma=jnp.float64(gamma_cost),
    )

    kmm = calc_kmm(tc, sp)
    gs_val = gammastar(tc, sp)
    phi0 = jnp.float64(kphio) * ftemp_kphio(tc, c4=False)
    Iabs = ppfd * fapar
    ca = co2 * sp * 1e-6

    par_photosynth = ParPhotosynth(
        kmm=kmm,
        gammastar=gs_val,
        phi0=phi0,
        Iabs=Iabs,
        ca=ca,
        patm=sp,
        delta=rdark_leaf,
    )

    par_env = ParEnv(
        viscosity_water=viscosity_h2o(tc, sp),
        density_water=density_h2o(tc, sp),
        patm=sp,
        tc=tc,
        vpd=vpd,
    )

    # Optimise
    log_jmax_opt, dpsi_opt = optimise_midterm_multi(
        psi_soil, par_cost, par_photosynth, par_plant, par_env,
    )

    # Evaluate at optimum
    profit = fn_profit(
        log_jmax_opt, dpsi_opt, psi_soil,
        par_cost, par_photosynth, par_plant, par_env,
        do_optim=False,
    )

    jmax = jnp.exp(log_jmax_opt)
    dpsi = dpsi_opt
    gs = calc_gs(dpsi, psi_soil, par_plant, par_env)
    ci, aj = calc_assim_light_limited(gs, jmax, par_photosynth)

    # Vcmax from assimilation (same formula as Fortran)
    vcmax = aj * (ci + par_photosynth.kmm) / (
        ci * (1.0 - par_photosynth.delta)
        - (par_photosynth.gammastar + par_photosynth.kmm * par_photosynth.delta)
    )

    chi = ci / par_photosynth.ca
    chi_jmax_lim = jnp.float64(0.0)

    return {
        "jmax": jmax,
        "dpsi": dpsi,
        "gs": gs,
        "aj": aj,
        "ci": ci,
        "chi": chi,
        "vcmax": vcmax,
        "profit": profit,
        "chi_jmax_lim": chi_jmax_lim,
    }
