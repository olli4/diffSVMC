"""
Leaf-level functions for the SpaFHy water-balance submodel.

All functions are written in pure JAX for automatic differentiation.
Port of water_mod.f90 from the Fortran SVMC model.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from typing import NamedTuple

_CONSTANTS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "svmc-ref" / "constants"
_WATER_CONSTANTS = json.loads((_CONSTANTS_DIR / "water.json").read_text())


# ── Parameter structures ──────────────────────────────────────────────


class SoilHydroParams(NamedTuple):
    """Van Genuchten soil hydraulic parameters (matches spafhy_para_type)."""
    watsat: jax.Array     # Saturated volumetric water content (m³/m³)
    watres: jax.Array     # Residual volumetric water content (m³/m³)
    alpha_van: jax.Array  # Van Genuchten alpha (1/kPa)
    n_van: jax.Array      # Van Genuchten n (pore size distribution)
    ksat: jax.Array       # Saturated hydraulic conductivity (m/s)


class AeroParams(NamedTuple):
    """Aerodynamic resistance parameters for SpaFHy.

    Matches spafhy_para_type fields used by the Fortran aerodynamics routine.
    """
    hc: jax.Array         # Canopy height (m)
    zmeas: jax.Array      # Measurement height above canopy (m)
    zground: jax.Array    # Ground reference height (m)
    zo_ground: jax.Array  # Ground roughness length (m)
    w_leaf: jax.Array     # Leaf width (m)


# ── Leaf functions ────────────────────────────────────────────────────


def e_sat(
    tc: jax.Array,
    patm: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Saturation vapor pressure, slope, and psychrometric constant.

    Fortran: e_sat(T, P, s, g, esat) in water_mod.f90
    Magnus formula.

    Args:
        tc: Temperature (°C)
        patm: Atmospheric pressure (Pa)

    Returns:
        (esat, s, gamma) — saturation VP (Pa), slope (Pa/K),
        psychrometric constant (Pa/K)
    """
    nt = 273.15
    cp = 1004.67

    lam = 1.0e3 * (3147.5 - 2.37 * (tc + nt))
    esat = 1.0e3 * 0.6112 * jnp.exp(17.67 * tc / (tc + 273.16 - 29.66))
    s = 17.502 * 240.97 * esat / (240.97 + tc) ** 2
    gamma = patm * cp / (0.622 * lam)
    return esat, s, gamma


def penman_monteith(
    ae: jax.Array,
    d: jax.Array,
    tc: jax.Array,
    gs: jax.Array,
    ga: jax.Array,
    patm: jax.Array,
) -> jax.Array:
    """Penman-Monteith latent heat flux (W/m²).

    Fortran: penman_monteith(AE, D, T, Gs, Ga, P) in water_mod.f90

    Args:
        ae: Available energy (W/m²)
        d: Vapor pressure deficit (Pa)
        tc: Temperature (°C)
        gs: Surface conductance (m/s)
        ga: Aerodynamic conductance (m/s)
        patm: Atmospheric pressure (Pa)

    Returns:
        Latent heat flux LE (W/m²), clamped >= 0
    """
    cp = 1004.67
    rho = 1.25

    _, s, g = e_sat(tc, patm)

    le = (s * ae + rho * cp * ga * d) / (s + g * (1.0 + ga / gs))
    return jnp.maximum(le, 0.0)


def soil_water_retention_curve(
    vol_liq: jax.Array,
    params: SoilHydroParams,
) -> jax.Array:
    """Soil water potential (MPa) from volumetric water content.

    Fortran: soil_water_retention_curve in water_mod.f90
    Van Genuchten model.
    """
    m = 1.0 - 1.0 / params.n_van
    eff_porosity = jnp.maximum(0.01, params.watsat)
    satfrac = jnp.clip(
        (vol_liq - params.watres) / (eff_porosity - params.watres),
        1e-6, 1.0 - 1e-6,
    )
    smp = -(1.0 / params.alpha_van) * (satfrac ** (-1.0 / m) - 1.0) ** (1.0 / params.n_van)
    return smp * 0.001  # → MPa


def soil_hydraulic_conductivity(
    vol_liq: jax.Array,
    params: SoilHydroParams,
) -> jax.Array:
    """Soil hydraulic conductivity (m/s) from volumetric water content.

    Fortran: soil_hydraulic_conductivity in water_mod.f90
    Mualem–Van Genuchten model.
    """
    m = 1.0 - 1.0 / params.n_van
    eff_porosity = jnp.maximum(0.01, params.watsat)
    satfrac = jnp.clip(
        (vol_liq - params.watres) / (eff_porosity - params.watres),
        1e-6, 1.0,
    )
    # At saturation the VG formula gives ksat analytically, but the
    # gradient has a singularity (0^(m-1) with m<1).  Use a safe value
    # for the formula branch so jnp.where produces finite gradients.
    not_saturated = satfrac < 1.0
    satfrac_safe = jnp.where(not_saturated, satfrac, 0.5)
    k_formula = params.ksat * jnp.sqrt(satfrac_safe) * (1.0 - (1.0 - satfrac_safe ** (1.0 / m)) ** m) ** 2
    k = jnp.where(not_saturated, k_formula, params.ksat)
    return k


def aerodynamics(
    lai: jax.Array,
    uo: jax.Array,
    params: AeroParams,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Aerodynamic resistances for canopy and soil layers.

    Fortran: aerodynamics in water_mod.f90
    References: Cammalleri et al. (2010), Massman (1987), Magnani et al. (1998)

    Args:
        lai: Leaf area index (m²/m²)
        uo: Mean wind speed at reference height (m/s)
        params: Canopy/aerodynamic parameters

    Returns:
        (ra, rb, ras, ustar, Uh, Ug) — resistances (s/m) and velocities (m/s)
    """
    kv = 0.4       # von Karman constant
    beta_aero = 285.0  # Campbell & Norman eq. (7.33) x 42.0 mol/m³
    eps = 1e-16

    zm1 = params.hc + params.zmeas
    zg1 = jnp.minimum(params.zground, 0.1 * params.hc)
    alpha1 = lai / 2.0         # wind attenuation coefficient (Yi 2008 eq. 23)
    d = 0.66 * params.hc       # displacement height
    zom = 0.123 * params.hc    # roughness length for momentum
    zov = 0.1 * zom            # scalar roughness length
    zosv = 0.1 * params.zo_ground  # soil scalar roughness length

    # Solve ustar and U(hc) from log-profile above canopy
    ustar = uo * kv / jnp.log((zm1 - d) / zom)
    Uh = ustar / kv * jnp.log((params.hc - d) / zom)

    # U(zg) from exponential wind profile
    zn = jnp.minimum(zg1 / params.hc, 1.0)
    Ug = Uh * jnp.exp(alpha1 * (zn - 1.0))

    # Canopy aerodynamic resistance (s/m) — Magnani et al. 1998 PCE eq. B1
    ra = 1.0 / (kv**2 * uo) * jnp.log((zm1 - d) / zom) * jnp.log((zm1 - d) / zov)

    # Boundary-layer resistance (s/m) — Magnani et al. 1998 PCE eq. B5
    rb = jnp.where(
        lai > eps,
        1.0 / lai * beta_aero * jnp.sqrt(
            params.w_leaf / Uh * (alpha1 / (1.0 - jnp.exp(-alpha1 / 2.0)))
        ),
        0.0,
    )

    # Soil aerodynamic resistance (s/m)
    ras = 1.0 / (kv**2 * Ug) * jnp.log(params.zground / params.zo_ground) * jnp.log(params.zground / zosv)

    ra = ra + rb

    return ra, rb, ras, ustar, Uh, Ug


def exponential_smooth_met(
    met_daily: jax.Array,
    met_rolling: jax.Array,
    met_ind: int,
) -> tuple[jax.Array, int]:
    """Exponential smoothing for scaling meteorological parameters.

    Fortran: exponential_smooth_met in wrapper_yasso.f90

    Args:
        met_daily: Current daily meteorological values [2].
        met_rolling: Previous rolling average values [2].
        met_ind: Step counter (1 on first call).

    Returns:
        (new_met_rolling, new_met_ind)
    """
    alpha_smooth1 = _WATER_CONSTANTS["alpha_smooth1"]
    alpha_smooth2 = _WATER_CONSTANTS["alpha_smooth2"]

    if met_ind == 1:
        new_rolling = met_daily.copy()
        new_ind = met_ind + 1
    else:
        r0 = alpha_smooth1 * met_daily[0] + (1.0 - alpha_smooth1) * met_rolling[0]
        r1 = alpha_smooth2 * met_daily[1] + (1.0 - alpha_smooth2) * met_rolling[1]
        new_rolling = jnp.array([r0, r1])
        new_ind = met_ind

    return new_rolling, new_ind
