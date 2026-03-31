"""P-Hydro solver: differentiable projected-Newton optimisation.

Uses a fixed-iteration projected Newton optimizer with vectorized
step-size selection, implemented via ``jax.lax.fori_loop`` and
``jax.vmap``.  XLA compiles this into a single fused GPU kernel with
no data-dependent control flow, replacing the earlier JAXopt L-BFGS-B
solver whose ``while_loop`` caused ~100× slowdowns on GPU due to
per-iteration kernel-launch and host-sync overhead.

For the 2-variable P-Hydro problem, Newton's method:
- Computes the exact 2×2 Hessian (cheap for n=2)
- Uses vectorized step-size candidates (vmap) instead of while_loop Armijo
- Converges in ~8 iterations, matching L-BFGS-B trajectory and basin

The backward pass uses implicit differentiation through the optimality
condition (``jax.custom_vjp``), keeping the full integration loop
end-to-end differentiable without unrolling solver iterations.

Port of optimise_midterm_multi and pmodel_hydraulics_numerical from
phydro_mod.f90 in the Fortran SVMC model.
"""

import functools

import jax
import jax.numpy as jnp

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
_LO = jnp.array([_LOG_JMAX_LO, _DPSI_LO], dtype=jnp.float64)
_HI = jnp.array([_LOG_JMAX_HI, _DPSI_HI], dtype=jnp.float64)

# Projected-Newton hyper-parameters
_MAXITER = 12
_HESS_REG = 1e-4   # Hessian regularisation for backward pass
# Candidate step sizes for vectorized line search (vmap, no while_loop)
_STEP_SIZES = jnp.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
_BOUND_TOL = 1e-10

# Adaptive Levenberg-Marquardt damping (inspired by dlm-js's natural
# optimizer): λ shrinks toward Newton on acceptance, grows toward
# gradient descent on rejection.
_LM_INIT = 1e-2
_LM_SHRINK = 0.5
_LM_GROW = 4.0
_LM_MIN = 1e-8
_LM_MAX = 1e6


def _objective(
    params: jax.Array,
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> jax.Array:
    """Negative profit (minimisation target)."""
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


def _newton_solve(
    psi_soil, par_cost, par_photosynth, par_plant, par_env,
):
    """Projected Newton with adaptive LM damping via lax.fori_loop.

    Solves (H + λI)δ = g at each step; λ shrinks on accepted steps
    (approaching pure Newton) and grows on rejected steps (approaching
    steepest descent).  This eliminates the stall-at-initial-guess
    failure mode of fixed-regularisation Newton.
    """

    def obj_at(x):
        return _objective(x, psi_soil, par_cost, par_photosynth,
                          par_plant, par_env)

    def body(_, state):
        x, lam = state
        f_cur = obj_at(x)
        g = jax.grad(obj_at)(x)
        H = jax.hessian(obj_at)(x) + lam * jnp.eye(2)
        d = jnp.linalg.solve(H, g)
        # Evaluate candidates at each step size (vectorized, no while_loop)
        candidates = jnp.clip(
            x[None, :] - _STEP_SIZES[:, None] * d[None, :], _LO, _HI
        )
        vals = jax.vmap(obj_at)(candidates)
        # Include current point to guarantee monotonic descent
        vals = jnp.concatenate([vals, f_cur[None]])
        candidates = jnp.concatenate([candidates, x[None, :]])

        best_idx = jnp.argmin(vals)
        x_new = candidates[best_idx]
        # Adapt damping: shrink if improved (more Newton-like),
        # grow if stalled (more gradient-descent-like)
        improved = vals[best_idx] < f_cur - 1e-14
        new_lam = jnp.where(
            improved,
            jnp.maximum(lam * _LM_SHRINK, _LM_MIN),
            jnp.minimum(lam * _LM_GROW, _LM_MAX),
        )
        return (x_new, new_lam)

    x_opt, _ = jax.lax.fori_loop(
        0, _MAXITER, body, (_X0, jnp.float64(_LM_INIT))
    )
    return x_opt


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def _solve_impl(psi_soil, par_cost, par_photosynth, par_plant, par_env):
    return _newton_solve(psi_soil, par_cost, par_photosynth,
                         par_plant, par_env)


def _solve_fwd(psi_soil, par_cost, par_photosynth, par_plant, par_env):
    x_opt = _solve_impl(psi_soil, par_cost, par_photosynth,
                        par_plant, par_env)
    return x_opt, (x_opt, psi_soil, par_cost, par_photosynth,
                   par_plant, par_env)


def _solve_bwd(res, g):
    """Implicit differentiation: dL/dθ = −(∂²f/∂x²)⁻¹ · ∂²f/∂x∂θ · v.

    Box constraints require an active-set-aware solve: coordinates pinned
    at an active bound have zero sensitivity, so only the free subspace
    participates in the implicit system.
    """
    x_opt, psi_soil, par_cost, par_photosynth, par_plant, par_env = res

    def obj_x(x):
        return _objective(x, psi_soil, par_cost, par_photosynth,
                          par_plant, par_env)

    grad_x = jax.grad(obj_x)(x_opt)
    active_lo = (jnp.abs(x_opt - _LO) <= _BOUND_TOL) & (grad_x >= 0.0)
    active_hi = (jnp.abs(x_opt - _HI) <= _BOUND_TOL) & (grad_x <= 0.0)
    free_mask = ~(active_lo | active_hi)
    free = free_mask.astype(jnp.float64)

    hess = jax.hessian(obj_x)(x_opt) + _HESS_REG * jnp.eye(2)
    masked_hess = free[:, None] * hess * free[None, :] + jnp.diag(1.0 - free)
    masked_g = g * free
    u = jnp.linalg.solve(masked_hess, masked_g) * free

    def grad_x_wrt_params(psi_soil, par_cost, par_photosynth, par_plant, par_env):
        return jax.grad(_objective)(
            x_opt, psi_soil, par_cost, par_photosynth, par_plant, par_env)

    _, vjp_fn = jax.vjp(grad_x_wrt_params, psi_soil, par_cost,
                        par_photosynth, par_plant, par_env)
    return vjp_fn(-u)


_solve_impl.defvjp(_solve_fwd, _solve_bwd)


@jax.jit
def optimise_midterm_multi(
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> tuple[jax.Array, jax.Array]:
    """Find optimal (log_jmax, dpsi) that maximise the profit function.

    Uses projected Newton with vectorized step-size selection via
    ``lax.fori_loop`` + ``vmap`` for GPU-friendly execution (single
    fused kernel, no while_loop sync overhead) and implicit
    differentiation through the optimality condition.

    Returns:
        (log_jmax_opt, dpsi_opt) — the optimised parameters
    """
    x_opt = _solve_impl(psi_soil, par_cost, par_photosynth, par_plant, par_env)
    return x_opt[0], x_opt[1]


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
