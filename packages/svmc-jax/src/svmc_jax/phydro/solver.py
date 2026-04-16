"""P-Hydro solver: differentiable projected optimisation.

Uses a fixed-budget projected limited-memory BFGS optimizer with static
memory, masked active-set handling, and vectorized trial step sizes via
``jax.lax.fori_loop`` and ``jax.vmap``. A short projected-Newton polish
then tightens the final optimum without reintroducing dynamic control
flow. XLA can still compile this into static control flow suitable for
GPU batching while avoiding the dynamic breakpoint sorting and line-
search loops that made classical traced L-BFGS-B slow on GPU.

The backward pass continues to use implicit differentiation through the
optimality condition (``jax.custom_vjp``), keeping the full integration
loop end-to-end differentiable without unrolling solver iterations.

Port of optimise_midterm_multi and pmodel_hydraulics_numerical from
phydro_mod.f90 in the Fortran SVMC model.
"""

import functools
from typing import Literal, NamedTuple

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
DEFAULT_PHYDRO_OPTIMIZER = "projected_lbfgs"
PhydroOptimizer = Literal["projected_lbfgs", "projected_newton"]

# Fortran box bounds for the optimisation variables
_LOG_JMAX_LO = -10.0
_LOG_JMAX_HI = 10.0
_DPSI_LO = 1.0e-4
_DPSI_HI = 1.0e6

_LO = jnp.array([_LOG_JMAX_LO, _DPSI_LO], dtype=jnp.float64)
_HI = jnp.array([_LOG_JMAX_HI, _DPSI_HI], dtype=jnp.float64)
_STARTS = jnp.array([
    [4.0, 1.0],
    [1.0, 0.05],
    [_LOG_JMAX_LO, _DPSI_LO],
], dtype=jnp.float64)

# Projected L-BFGS hyper-parameters
_MAXITER = 16
_STEP_SIZES = jnp.array([
    1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3,
    1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1,
    1.0, 3.0, 10.0,
], dtype=jnp.float64)
_BOUND_TOL = 1e-10
_HESS_REG = 1e-4   # Hessian regularisation for backward pass
_MEMORY = 4
_CURV_EPS = 1e-12
_POLISH_ITERS = 4

# Projected-Newton baseline hyper-parameters
_NEWTON_X0 = jnp.array([4.0, 1.0], dtype=jnp.float64)
_NEWTON_MAXITER = 12
_NEWTON_STEP_SIZES = jnp.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0], dtype=jnp.float64)
_LM_INIT = 1e-2
_LM_SHRINK = 0.5
_LM_GROW = 4.0
_LM_MIN = 1e-8
_LM_MAX = 1e6


class PModelHydraulicsResult(NamedTuple):
    """Typed P-Hydro result container.

    NamedTuple keeps the JAX pytree structure smaller than a dict in the
    traced integration loop, while the string-key fallback preserves the
    existing test and caller surface.
    """

    jmax: jax.Array
    dpsi: jax.Array
    gs: jax.Array
    aj: jax.Array
    ci: jax.Array
    chi: jax.Array
    vcmax: jax.Array
    profit: jax.Array
    chi_jmax_lim: jax.Array

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        return tuple.__getitem__(self, item)


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


def _free_mask(x: jax.Array, grad_x: jax.Array) -> jax.Array:
    active_lo = (jnp.abs(x - _LO) <= _BOUND_TOL) & (grad_x >= 0.0)
    active_hi = (jnp.abs(x - _HI) <= _BOUND_TOL) & (grad_x <= 0.0)
    return ~(active_lo | active_hi)


def _projected_newton_solve(
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> jax.Array:
    """Projected Newton baseline with adaptive LM damping."""

    def obj_at(x):
        return _objective(x, psi_soil, par_cost, par_photosynth,
                          par_plant, par_env)

    f0, g0 = jax.value_and_grad(obj_at)(_NEWTON_X0)

    def body(_, state):
        x, f_cur, g_cur, lam = state
        hess = jax.hessian(obj_at)(x) + lam * jnp.eye(2)
        step = jnp.linalg.solve(hess, g_cur)
        candidates = jnp.clip(
            x[None, :] - _NEWTON_STEP_SIZES[:, None] * step[None, :], _LO, _HI
        )
        vals = jax.vmap(obj_at)(candidates)
        finite = jnp.isfinite(vals) & jnp.all(jnp.isfinite(candidates), axis=1)
        safe_vals = jnp.where(finite, vals, jnp.inf)

        best_idx = jnp.argmin(safe_vals)
        best_val = safe_vals[best_idx]
        improved = jnp.isfinite(best_val) & (best_val < f_cur - 1e-14)
        x_next = jnp.where(improved, candidates[best_idx], x)
        f_next = jnp.where(improved, best_val, f_cur)
        # lax.cond: on CPU, skips the gradient when the step was rejected;
        # on GPU both branches execute anyway (SIMT), so this is harmless.
        g_next = jax.lax.cond(
            improved,
            lambda: jax.grad(obj_at)(x_next),
            lambda: g_cur,
        )
        new_lam = jnp.where(
            improved,
            jnp.maximum(lam * _LM_SHRINK, _LM_MIN),
            jnp.minimum(lam * _LM_GROW, _LM_MAX),
        )
        return x_next, f_next, g_next, new_lam

    x_opt, _, _, _ = jax.lax.fori_loop(
        0, _NEWTON_MAXITER, body,
        (_NEWTON_X0, f0, g0, jnp.float64(_LM_INIT)),
    )
    return x_opt


def _lbfgs_direction(
    grad_x: jax.Array,
    s_hist: jax.Array,
    y_hist: jax.Array,
    rho_hist: jax.Array,
    valid_hist: jax.Array,
    free: jax.Array,
) -> jax.Array:
    """Return a projected L-BFGS descent direction via static two-loop recursion."""
    q0 = grad_x * free
    alpha0 = jnp.zeros((_MEMORY,), dtype=jnp.float64)

    def backward_body(i, state):
        q, alpha = state
        idx = _MEMORY - 1 - i
        valid = valid_hist[idx]
        s_i = s_hist[idx] * free
        y_i = y_hist[idx] * free
        rho_i = jnp.where(valid, rho_hist[idx], 0.0)
        alpha_i = rho_i * jnp.dot(s_i, q)
        q_next = q - alpha_i * y_i
        alpha = alpha.at[idx].set(jnp.where(valid, alpha_i, 0.0))
        return q_next, alpha

    q, alpha = jax.lax.fori_loop(0, _MEMORY, backward_body, (q0, alpha0))

    valid_idx = jnp.where(valid_hist, jnp.arange(_MEMORY) + 1, 0)
    latest_idx = jnp.maximum(jnp.max(valid_idx) - 1, 0)
    latest_valid = valid_hist[latest_idx]
    y_last = y_hist[latest_idx] * free
    s_last = s_hist[latest_idx] * free
    yy = jnp.dot(y_last, y_last)
    sy = jnp.dot(s_last, y_last)
    gamma = jnp.where(
        latest_valid & (yy > _CURV_EPS) & (sy > _CURV_EPS),
        sy / yy,
        1.0,
    )
    r0 = gamma * q

    def forward_body(i, r):
        valid = valid_hist[i]
        s_i = s_hist[i] * free
        y_i = y_hist[i] * free
        rho_i = jnp.where(valid, rho_hist[i], 0.0)
        beta = rho_i * jnp.dot(y_i, r)
        return r + (alpha[i] - beta) * s_i

    r = jax.lax.fori_loop(0, _MEMORY, forward_body, r0)
    direction = -r * free
    fallback = -q0
    use_fallback = (
        (jnp.linalg.norm(direction) <= _CURV_EPS)
        | (jnp.dot(direction, q0) >= 0.0)
        | (jnp.linalg.norm(q0) <= _CURV_EPS)
    )
    return jnp.where(use_fallback, fallback, direction)


def _update_memory(
    s_hist: jax.Array,
    y_hist: jax.Array,
    rho_hist: jax.Array,
    valid_hist: jax.Array,
    step: jax.Array,
    y_vec: jax.Array,
    free: jax.Array,
    accept: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    step = step * free
    y_vec = y_vec * free
    sy = jnp.dot(step, y_vec)
    accept_update = accept & jnp.isfinite(sy) & (sy > _CURV_EPS)
    rho = jnp.where(accept_update, 1.0 / sy, 0.0)
    s_shifted = jnp.concatenate([s_hist[1:], step[None, :]], axis=0)
    y_shifted = jnp.concatenate([y_hist[1:], y_vec[None, :]], axis=0)
    rho_shifted = jnp.concatenate([rho_hist[1:], rho[None]], axis=0)
    valid_shifted = jnp.concatenate(
        [valid_hist[1:], jnp.array([accept_update], dtype=jnp.bool_)], axis=0
    )
    s_hist = jnp.where(accept_update, s_shifted, s_hist)
    y_hist = jnp.where(accept_update, y_shifted, y_hist)
    rho_hist = jnp.where(accept_update, rho_shifted, rho_hist)
    valid_hist = jnp.where(accept_update, valid_shifted, valid_hist)
    return s_hist, y_hist, rho_hist, valid_hist


def _lbfgs_solve_from_start(
    start: jax.Array,
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> jax.Array:
    """Projected L-BFGS with fixed memory and static trial step sizes."""

    def obj_at(x):
        return _objective(x, psi_soil, par_cost, par_photosynth,
                          par_plant, par_env)

    # Pre-compute initial objective and gradient; carried across iterations
    # so that each iteration avoids redundant recomputation.
    f0, g0 = jax.value_and_grad(obj_at)(start)

    init_state = (
        start,
        f0,
        g0,
        jnp.zeros((_MEMORY, 2), dtype=jnp.float64),
        jnp.zeros((_MEMORY, 2), dtype=jnp.float64),
        jnp.zeros((_MEMORY,), dtype=jnp.float64),
        jnp.zeros((_MEMORY,), dtype=jnp.bool_),
    )

    def body(_, state):
        x, f_cur, g_cur, s_hist, y_hist, rho_hist, valid_hist = state
        free = _free_mask(x, g_cur).astype(jnp.float64)
        direction = _lbfgs_direction(
            g_cur, s_hist, y_hist, rho_hist, valid_hist, free,
        )
        candidates = jnp.clip(
            x[None, :] + _STEP_SIZES[:, None] * direction[None, :], _LO, _HI
        )
        vals = jax.vmap(obj_at)(candidates)
        finite = jnp.isfinite(vals) & jnp.all(jnp.isfinite(candidates), axis=1)
        safe_vals = jnp.where(finite, vals, jnp.inf)
        best_idx = jnp.argmin(safe_vals)
        best_val = safe_vals[best_idx]
        improved = jnp.isfinite(best_val) & (best_val < f_cur - 1e-14)
        x_next = jnp.where(improved, candidates[best_idx], x)
        f_next = jnp.where(improved, best_val, f_cur)
        g_next = jax.lax.cond(
            improved,
            lambda: jax.grad(obj_at)(x_next),
            lambda: g_cur,
        )
        s_hist, y_hist, rho_hist, valid_hist = _update_memory(
            s_hist,
            y_hist,
            rho_hist,
            valid_hist,
            x_next - x,
            g_next - g_cur,
            free,
            improved,
        )
        return (x_next, f_next, g_next, s_hist, y_hist, rho_hist, valid_hist)

    x_opt, _, _, _, _, _, _ = jax.lax.fori_loop(0, _MAXITER, body, init_state)
    return x_opt


def _projected_newton_polish(
    start: jax.Array,
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> jax.Array:
    """Run a few static projected-Newton refinement steps."""

    def obj_at(x):
        return _objective(x, psi_soil, par_cost, par_photosynth,
                          par_plant, par_env)

    f0, g0 = jax.value_and_grad(obj_at)(start)

    def body(_, state):
        x, f_cur, g_cur = state
        free = _free_mask(x, g_cur).astype(jnp.float64)
        hess = jax.hessian(obj_at)(x) + _HESS_REG * jnp.eye(2)
        masked_hess = free[:, None] * hess * free[None, :] + jnp.diag(1.0 - free)
        newton_dir = -jnp.linalg.solve(masked_hess, g_cur * free) * free
        grad_dir = -(g_cur * free)
        newton_ok = jnp.all(jnp.isfinite(newton_dir)) & (jnp.dot(newton_dir, g_cur * free) < 0.0)
        direction = jnp.where(newton_ok, newton_dir, grad_dir)
        candidates = jnp.clip(
            x[None, :] + _STEP_SIZES[:, None] * direction[None, :], _LO, _HI
        )
        vals = jax.vmap(obj_at)(candidates)
        finite = jnp.isfinite(vals) & jnp.all(jnp.isfinite(candidates), axis=1)
        safe_vals = jnp.where(finite, vals, jnp.inf)
        best_idx = jnp.argmin(safe_vals)
        best_val = safe_vals[best_idx]
        improved = jnp.isfinite(best_val) & (best_val < f_cur - 1e-14)
        x_next = jnp.where(improved, candidates[best_idx], x)
        f_next = jnp.where(improved, best_val, f_cur)
        g_next = jax.lax.cond(
            improved,
            lambda: jax.grad(obj_at)(x_next),
            lambda: g_cur,
        )
        return (x_next, f_next, g_next)

    x_opt, _, _ = jax.lax.fori_loop(0, _POLISH_ITERS, body, (start, f0, g0))
    return x_opt


def _lbfgs_solve(
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> jax.Array:
    """Static multi-start projected L-BFGS, selecting the best final objective."""

    def solve_one(start):
        x_lbfgs = _lbfgs_solve_from_start(
            start, psi_soil, par_cost, par_photosynth, par_plant, par_env,
        )
        return _projected_newton_polish(
            x_lbfgs, psi_soil, par_cost, par_photosynth, par_plant, par_env,
        )

    candidates = jax.vmap(solve_one)(_STARTS)
    vals = jax.vmap(
        lambda x: _objective(x, psi_soil, par_cost, par_photosynth,
                             par_plant, par_env)
    )(candidates)
    finite = jnp.isfinite(vals) & jnp.all(jnp.isfinite(candidates), axis=1)
    safe_vals = jnp.where(finite, vals, jnp.inf)
    return candidates[jnp.argmin(safe_vals)]


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def _solve_impl_lbfgs(psi_soil, par_cost, par_photosynth, par_plant, par_env):
    return _lbfgs_solve(psi_soil, par_cost, par_photosynth,
                        par_plant, par_env)


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def _solve_impl_newton(psi_soil, par_cost, par_photosynth, par_plant, par_env):
    return _projected_newton_solve(
        psi_soil, par_cost, par_photosynth, par_plant, par_env,
    )


def _solve_fwd_lbfgs(psi_soil, par_cost, par_photosynth, par_plant, par_env):
    x_opt = _solve_impl_lbfgs(psi_soil, par_cost, par_photosynth,
                              par_plant, par_env)
    return x_opt, (x_opt, psi_soil, par_cost, par_photosynth,
                   par_plant, par_env)


def _solve_fwd_newton(psi_soil, par_cost, par_photosynth, par_plant, par_env):
    x_opt = _solve_impl_newton(psi_soil, par_cost, par_photosynth,
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
    free_mask = _free_mask(x_opt, grad_x)
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


_solve_impl_lbfgs.defvjp(_solve_fwd_lbfgs, _solve_bwd)
_solve_impl_newton.defvjp(_solve_fwd_newton, _solve_bwd)


def _solve_selected(
    solver_kind: PhydroOptimizer,
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
) -> jax.Array:
    if solver_kind == "projected_lbfgs":
        return _solve_impl_lbfgs(
            psi_soil, par_cost, par_photosynth, par_plant, par_env,
        )
    if solver_kind == "projected_newton":
        return _solve_impl_newton(
            psi_soil, par_cost, par_photosynth, par_plant, par_env,
        )
    raise ValueError(
        "Unknown phydro optimizer "
        f"{solver_kind!r}; expected 'projected_lbfgs' or 'projected_newton'"
    )


@functools.partial(jax.jit, static_argnames=("solver_kind",))
def optimise_midterm_multi(
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
    solver_kind: PhydroOptimizer = DEFAULT_PHYDRO_OPTIMIZER,
) -> tuple[jax.Array, jax.Array]:
    """Find optimal (log_jmax, dpsi) that maximise the profit function.

    Supports two static solver choices:
    - ``projected_lbfgs``: three-start projected L-BFGS with Newton polish
    - ``projected_newton``: the earlier adaptive-LM projected-Newton path

    Both remain implicitly differentiated through the optimality
    condition.

    Returns:
        (log_jmax_opt, dpsi_opt) — the optimised parameters
    """
    x_opt = _solve_selected(
        solver_kind, psi_soil, par_cost, par_photosynth, par_plant, par_env,
    )
    return x_opt[0], x_opt[1]


@functools.partial(jax.jit, static_argnames=("solver_kind",))
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
    solver_kind: PhydroOptimizer = DEFAULT_PHYDRO_OPTIMIZER,
) -> PModelHydraulicsResult:
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

    log_jmax_opt, dpsi_opt = optimise_midterm_multi(
        psi_soil, par_cost, par_photosynth, par_plant, par_env,
        solver_kind=solver_kind,
    )

    profit = fn_profit(
        log_jmax_opt, dpsi_opt, psi_soil,
        par_cost, par_photosynth, par_plant, par_env,
        do_optim=False,
    )

    jmax = jnp.exp(log_jmax_opt)
    dpsi = dpsi_opt
    gs = calc_gs(dpsi, psi_soil, par_plant, par_env)
    ci, aj = calc_assim_light_limited(gs, jmax, par_photosynth)

    vcmax = aj * (ci + par_photosynth.kmm) / (
        ci * (1.0 - par_photosynth.delta)
        - (par_photosynth.gammastar + par_photosynth.kmm * par_photosynth.delta)
    )

    chi = ci / par_photosynth.ca
    chi_jmax_lim = jnp.float64(0.0)

    return PModelHydraulicsResult(
        jmax=jmax,
        dpsi=dpsi,
        gs=gs,
        aj=aj,
        ci=ci,
        chi=chi,
        vcmax=vcmax,
        profit=profit,
        chi_jmax_lim=chi_jmax_lim,
    )
