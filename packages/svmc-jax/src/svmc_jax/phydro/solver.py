"""
P-Hydro solver: differentiable optimization using JAX.

Replaces Fortran's finite-difference L-BFGS-B with JAX autodiff gradients
through fn_profit. Uses scipy.optimize.minimize (L-BFGS-B) with jax.grad
for exact gradients.

Port of optimise_midterm_multi and pmodel_hydraulics_numerical from
phydro_mod.f90 in the Fortran SVMC model.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

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
MAX_OPT_STEPS = 512
OPT_LEARNING_RATE = 0.05
OPT_CLIP_NORM = 10.0


class OptimParams(NamedTuple):
    """Optimized parameter pair for the P-Hydro profit objective."""

    log_jmax: jax.Array
    dpsi: jax.Array


class OptimCarry(NamedTuple):
    """Loop state for the projected Optax solver."""

    params: OptimParams
    opt_state: optax.OptState
    best_params: OptimParams
    best_loss: jax.Array


OPTIMIZER = optax.chain(
    optax.clip_by_global_norm(OPT_CLIP_NORM),
    optax.adam(OPT_LEARNING_RATE),
)


def _project_params(params: OptimParams) -> OptimParams:
    """Project optimizer parameters back into the Fortran box bounds."""

    return OptimParams(
        log_jmax=jnp.clip(params.log_jmax, -10.0, 10.0),
        dpsi=jnp.clip(params.dpsi, 1e-4, 1e6),
    )


def _select_params(
    predicate: jax.Array,
    when_true: OptimParams,
    when_false: OptimParams,
) -> OptimParams:
    """Select one of two parameter states without leaving traced JAX code."""

    return OptimParams(
        log_jmax=jnp.where(predicate, when_true.log_jmax, when_false.log_jmax),
        dpsi=jnp.where(predicate, when_true.dpsi, when_false.dpsi),
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

    Uses a projected Optax Adam loop, keeping the entire optimization in
    traced JAX code so it remains safe to JIT and compose inside larger
    simulation and calibration loops.

    Returns:
        (log_jmax_opt, dpsi_opt) — the optimised parameters
    """
    def objective(params: OptimParams) -> jax.Array:
        return fn_profit(
            params.log_jmax,
            params.dpsi,
            psi_soil, par_cost, par_photosynth, par_plant, par_env,
            do_optim=True,
    )

    objective_and_grad = jax.value_and_grad(objective)
    init_params = OptimParams(
        log_jmax=jnp.float64(4.0),
        dpsi=jnp.float64(1.0),
    )
    init_carry = OptimCarry(
        params=init_params,
        opt_state=OPTIMIZER.init(init_params),
        best_params=init_params,
        best_loss=jnp.float64(jnp.inf),
    )

    def step(_, carry: OptimCarry) -> OptimCarry:
        loss, grads = objective_and_grad(carry.params)
        better = loss < carry.best_loss
        best_params = _select_params(better, carry.params, carry.best_params)
        best_loss = jnp.where(better, loss, carry.best_loss)
        updates, opt_state = OPTIMIZER.update(grads, carry.opt_state, carry.params)
        params = _project_params(optax.apply_updates(carry.params, updates))
        return OptimCarry(
            params=params,
            opt_state=opt_state,
            best_params=best_params,
            best_loss=best_loss,
        )

    final_carry = jax.lax.fori_loop(0, MAX_OPT_STEPS, step, init_carry)
    final_loss = objective(final_carry.params)
    better = final_loss < final_carry.best_loss
    best_params = _select_params(better, final_carry.params, final_carry.best_params)
    return best_params.log_jmax, best_params.dpsi


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
