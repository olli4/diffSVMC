"""
Leaf-level functions for the P-Hydro submodel.

All functions are written in pure JAX for automatic differentiation.
Port of phydro_mod.f90 from the Fortran SVMC model.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


_VISCOSITY_H = jnp.array([
    [0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0],
    [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573],
    [-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0],
    [0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0],
    [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0],
    [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264],
], dtype=jnp.float64)


# ── Parameter structures ──────────────────────────────────────────────


class ParEnv(NamedTuple):
    """Environmental parameters."""
    viscosity_water: jax.Array
    density_water: jax.Array
    patm: jax.Array
    tc: jax.Array
    vpd: jax.Array


class ParPlant(NamedTuple):
    """Plant hydraulic parameters."""
    conductivity: jax.Array  # Leaf conductivity (m)
    psi50: jax.Array         # Water potential at 50% loss of conductivity (MPa)
    b: jax.Array             # Shape parameter for vulnerability curve


class ParCost(NamedTuple):
    """Cost parameters for profit optimization."""
    alpha: jax.Array  # Cost of Jmax
    gamma: jax.Array  # Cost of hydraulic repair


class ParPhotosynth(NamedTuple):
    """Photosynthesis parameters."""
    kmm: jax.Array        # Michaelis-Menten coefficient (Pa)
    gammastar: jax.Array  # CO2 compensation point (Pa)
    phi0: jax.Array       # Quantum yield efficiency
    Iabs: jax.Array       # Absorbed PAR
    ca: jax.Array         # Ambient CO2 partial pressure (Pa)
    patm: jax.Array       # Atmospheric pressure (Pa)
    delta: jax.Array      # Dark respiration fraction


# ── Leaf functions ────────────────────────────────────────────────────


def ftemp_arrh(tk: jax.Array, dha: jax.Array) -> jax.Array:
    """Arrhenius-type temperature response.

    Fortran: ftemp_arrh in phydro_mod.f90
    """
    kR = 8.3145
    tkref = 298.15
    return jnp.exp(dha * (tk - tkref) / (tkref * kR * tk))


def gammastar(tc: jax.Array, patm: jax.Array) -> jax.Array:
    """Photorespiratory CO2 compensation point Γ* (Pa).

    Fortran: gammastar in phydro_mod.f90
    References: Bernacchi et al. (2001)
    """
    dha = 37830.0
    gs25_0 = 4.332
    patm0 = 101325.0
    tk = tc + 273.15
    return gs25_0 * patm / patm0 * ftemp_arrh(tk, dha)


def ftemp_kphio(tc: jax.Array, c4: bool = False) -> jax.Array:
    """Temperature dependence of quantum yield efficiency.

    Fortran: ftemp_kphio in phydro_mod.f90
    References: Bernacchi et al. (2003)
    """
    if c4:
        raw = -0.064 + 0.03 * tc - 0.000464 * tc**2
    else:
        raw = 0.352 + 0.022 * tc - 3.4e-4 * tc**2
    return jnp.maximum(raw, 0.0)


def density_h2o(tc: jax.Array, patm: jax.Array) -> jax.Array:
    """Density of water (kg/m³) via the Tumlirz Equation.

    Fortran: density_h2o in phydro_mod.f90
    References: Fisher & Dial (1975)
    """
    tc2 = tc * tc
    tc3 = tc2 * tc
    tc4 = tc3 * tc
    tc5 = tc4 * tc
    tc6 = tc5 * tc
    tc7 = tc6 * tc
    tc8 = tc7 * tc
    tc9 = tc8 * tc

    lambda_ = (1788.316 + 21.55053 * tc - 0.4695911 * tc2
               + 3.096363e-3 * tc3 - 7.341182e-6 * tc4)
    po = (5918.499 + 58.05267 * tc - 1.1253317 * tc2
          + 6.6123869e-3 * tc3 - 1.4661625e-5 * tc4)
    vinf = (0.6980547 - 7.435626e-4 * tc + 3.704258e-5 * tc2
            - 6.315724e-7 * tc3 + 9.829576e-9 * tc4
            - 1.197269e-10 * tc5 + 1.005461e-12 * tc6
            - 5.437898e-15 * tc7 + 1.69946e-17 * tc8
            - 2.295063e-20 * tc9)

    pbar = 1e-5 * patm
    v = vinf + lambda_ / (po + pbar)
    return 1e3 / v


def viscosity_h2o(tc: jax.Array, patm: jax.Array) -> jax.Array:
    """Viscosity of water (Pa·s).

    Fortran: viscosity_h2o in phydro_mod.f90
    References: Huber et al. (2009)
    """
    tk_ast = 647.096
    rho_ast = 322.0
    mu_ast = 1e-6

    rho = density_h2o(tc, patm)
    tbar = (tc + 273.15) / tk_ast
    rbar = rho / rho_ast

    # mu0 (Eq. 11 & Table 2)
    mu0_denom = 1.67752 + 2.20462 / tbar + 0.6366564 / tbar**2 - 0.241605 / tbar**3
    mu0 = 1e2 * jnp.sqrt(tbar) / mu0_denom

    ctbar = 1.0 / tbar - 1.0
    rbar_m1 = rbar - 1.0

    # Vectorised mu1 sum — replaces nested fori_loops with array ops.
    # Power vectors via sequential cumprod (matches original loop
    # accumulation order for floating-point reproducibility).
    _one64 = jnp.ones(1, dtype=jnp.float64)
    rbar_pows = jnp.cumprod(jnp.concatenate([_one64, jnp.full(6, rbar_m1)]))
    ctbar_pows = jnp.cumprod(jnp.concatenate([_one64, jnp.full(5, ctbar)]))
    # coefs[i] = sum_j H[j,i] * rbar_m1^j   (inner polynomial per column)
    coefs = _VISCOSITY_H.T @ rbar_pows          # (6,)
    mu1_sum = jnp.dot(ctbar_pows, coefs)

    mu1 = jnp.exp(rbar * mu1_sum)
    return mu0 * mu1 * mu_ast


def calc_kmm(tc: jax.Array, patm: jax.Array) -> jax.Array:
    """Michaelis-Menten coefficient for Rubisco-limited photosynthesis (Pa).

    Fortran: calc_kmm in phydro_mod.f90
    """
    dhac = 79430.0
    dhao = 36380.0
    kco = 2.09476e5
    kc25 = 39.97
    ko25 = 27480.0

    tk = tc + 273.15
    kc = kc25 * ftemp_arrh(tk, dhac)
    ko = ko25 * ftemp_arrh(tk, dhao)
    po = kco * 1e-6 * patm  # O2 partial pressure
    return kc * (1.0 + po / ko)


def scale_conductivity(K: jax.Array, par_env: ParEnv) -> jax.Array:
    """Scale plant conductivity to mol/m²/s/MPa.

    Fortran: scale_conductivity in phydro_mod.f90
    """
    mol_h2o_per_kg = 55.5
    K2 = K / par_env.viscosity_water
    K3 = K2 * par_env.density_water * mol_h2o_per_kg
    return K3 * 1e6


def calc_gs(dpsi: jax.Array, psi_soil: jax.Array,
            par_plant: ParPlant, par_env: ParEnv) -> jax.Array:
    """Stomatal conductance (mol/m²/s).

    Fortran: calc_gs in phydro_mod.f90
    Uses the approximate integral of the vulnerability curve.
    """
    K = scale_conductivity(par_plant.conductivity, par_env)
    D = par_env.vpd / par_env.patm
    psi_mid = psi_soil - dpsi / 2.0
    papprox = 0.5 ** ((psi_mid / par_plant.psi50) ** par_plant.b)
    return K / 1.6 / D * dpsi * papprox


def calc_assim_light_limited(
    gs: jax.Array, jmax: jax.Array, par: ParPhotosynth
) -> tuple[jax.Array, jax.Array]:
    """Electron-transport-limited assimilation.

    Fortran: calc_assim_light_limited in phydro_mod.f90

    Returns:
        (ci, aj) — leaf internal CO2 (Pa), assimilation rate (µmol/m²/s)
    """
    ca = par.ca
    gs0 = gs * 1e6 / par.patm

    phi0iabs = par.phi0 * par.Iabs
    jlim = phi0iabs / jnp.sqrt(1.0 + (4.0 * phi0iabs / jmax) ** 2)
    d = par.delta

    # Quadratic: A*ci² + B*ci + C = 0
    A = -gs0
    B = gs0 * ca - gs0 * 2.0 * par.gammastar - jlim * (1.0 - d)
    C = gs0 * ca * 2.0 * par.gammastar + jlim * (par.gammastar + d * par.kmm)

    disc = jnp.maximum(B * B - 4.0 * A * C, 0.0)
    q = -0.5 * (B + jnp.sqrt(disc))
    ci = q / A
    aj = gs0 * (ca - ci)
    return ci, aj


def fn_profit(
    log_jmax: jax.Array,
    dpsi: jax.Array,
    psi_soil: jax.Array,
    par_cost: ParCost,
    par_photosynth: ParPhotosynth,
    par_plant: ParPlant,
    par_env: ParEnv,
    hypothesis: str = "PM",
    do_optim: bool = False,
) -> jax.Array:
    """Profit function for P-Hydro optimization.

    Fortran: fn_profit in phydro_mod.f90

    profit = Aj − α·Jmax − γ·Δψ²  (PM)
    """
    jmax = jnp.exp(log_jmax)
    gs = calc_gs(dpsi, psi_soil, par_plant, par_env)
    _ci, aj = calc_assim_light_limited(gs, jmax, par_photosynth)

    costs = par_cost.alpha * jmax + par_cost.gamma * dpsi ** 2

    if hypothesis == "PM":
        profit = aj - costs
    else:
        profit = -costs / (aj + 1e-4)

    if do_optim:
        return -profit
    return profit


def quadratic(a: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """Solve quadratic equation a·x² + b·x + c = 0 for root r1.

    Fortran: quadratic in phydro_mod.f90
    Uses the Numerical Recipes formula q = -0.5·(b + √(b²−4ac)), r1 = q/a.
    Edge cases: a=0,b≠0 → -c/b; a=0,b=0 → 0.
    """
    disc = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    q = -0.5 * (b + jnp.sqrt(disc))
    # a == 0: linear or degenerate
    r1_quadratic = q / a
    r1_linear = jnp.where(b == 0.0, 0.0, -c / b)
    return jnp.where(a == 0.0, r1_linear, r1_quadratic)
