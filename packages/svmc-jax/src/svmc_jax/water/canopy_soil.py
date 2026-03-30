"""Phase 3 SpaFHy composition functions: canopy water balance and soil water balance.

Port of canopy_water_snow, ground_evaporation, canopy_water_flux, and soil_water
from water_mod.f90 in the Fortran SVMC model.

All functions are written in pure JAX for automatic differentiation.
Branches use jnp.where for autodiff-friendly control flow.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from .leaf_functions import (
    penman_monteith,
    aerodynamics,
    soil_water_retention_curve,
    soil_hydraulic_conductivity,
    AeroParams,
    SoilHydroParams,
)


# ── State & parameter structures ──────────────────────────────────────


class CanopyWaterState(NamedTuple):
    """Canopy water and snow state variables."""
    CanopyStorage: jax.Array  # canopy water storage (mm)
    SWE: jax.Array            # snow water equivalent (mm)
    swe_i: jax.Array          # snow water equivalent as ice (mm)
    swe_l: jax.Array          # snow water equivalent as liquid (mm)


class CanopySnowParams(NamedTuple):
    """Parameters for canopy water interception and snow dynamics."""
    wmax: jax.Array         # storage capacity for rain (mm/LAI)
    wmaxsnow: jax.Array     # storage capacity for snow (mm/LAI)
    kmelt: jax.Array        # melt coefficient (mm/s)
    kfreeze: jax.Array      # freezing coefficient (mm/s)
    frac_snowliq: jax.Array # max fraction of liquid in snow (-)
    gsoil: jax.Array        # soil surface conductance (m/s)


class CanopySnowFlux(NamedTuple):
    """Flux outputs from canopy_water_snow."""
    Throughfall: jax.Array     # throughfall to soil / snowpack (mm)
    Interception: jax.Array    # canopy interception (mm)
    CanopyEvap: jax.Array      # evaporation / sublimation from canopy (mm)
    Unloading: jax.Array       # unloading from canopy (mm)
    PotInfiltration: jax.Array # potential infiltration to soil (mm)
    Melt: jax.Array            # snowmelt (mm)
    Freeze: jax.Array          # refreeze of liquid water in snow (mm)
    mbe: jax.Array             # mass-balance error (mm)


class CanopyWaterFlux(NamedTuple):
    """Full flux outputs from canopy_water_flux orchestrator."""
    Throughfall: jax.Array
    Interception: jax.Array
    CanopyEvap: jax.Array
    Unloading: jax.Array
    SoilEvap: jax.Array
    ET: jax.Array
    Transpiration: jax.Array
    PotInfiltration: jax.Array
    Melt: jax.Array
    Freeze: jax.Array
    mbe: jax.Array


class SoilWaterState(NamedTuple):
    """Soil water state variables."""
    WatSto: jax.Array      # root zone storage (mm)
    PondSto: jax.Array     # pond storage (mm)
    MaxWatSto: jax.Array   # root zone storage capacity (mm)
    MaxPondSto: jax.Array  # pond storage capacity (mm)
    FcSto: jax.Array       # field capacity storage (mm)
    Wliq: jax.Array        # volumetric water content (m³/m³)
    Psi: jax.Array         # water potential (MPa)
    Sat: jax.Array         # saturation ratio (-)
    Kh: jax.Array          # hydraulic conductivity (m/s)
    beta: jax.Array        # soil evaporation modifier (-)


class SoilWaterFlux(NamedTuple):
    """Soil water flux outputs."""
    Infiltration: jax.Array  # total inflow to root zone (mm)
    Runoff: jax.Array        # surface runoff (mm)
    Drainage: jax.Array      # drainage from bucket (mm)
    LateralFlow: jax.Array   # lateral flow (mm)
    ET: jax.Array            # evapotranspiration (mm)
    mbe: jax.Array           # mass-balance error (mm)


# ── Functions ─────────────────────────────────────────────────────────


def ground_evaporation(
    tc: jax.Array,
    ae: jax.Array,
    vpd: jax.Array,
    ras: jax.Array,
    patm: jax.Array,
    swe: jax.Array,
    beta: jax.Array,
    wat_sto: jax.Array,
    gsoil: jax.Array,
    time_step: jax.Array,
) -> jax.Array:
    """Evaporation from top soil layer (mm).

    Fortran: ground_evaporation in water_mod.f90

    Args:
        tc: Air temperature (°C)
        ae: Available energy (W/m²)
        vpd: Vapor pressure deficit (Pa)
        ras: Ground aerodynamic resistance (s/m)
        patm: Atmospheric pressure (Pa)
        swe: Snow water equivalent (mm) — evap zeroed when snow on ground
        beta: Soil evaporation modifier (-)
        wat_sto: Soil water storage (mm) — caps evaporation
        gsoil: Soil surface conductance (m/s)
        time_step: Timestep (hours)

    Returns:
        SoilEvap (mm during timestep)
    """
    eps = 1e-16
    lv = 1.0e3 * (3147.5 - 2.37 * (tc + 273.15))
    ga_s = 1.0 / ras

    erate = (time_step * 3600.0) * beta * penman_monteith(ae, vpd, tc, gsoil, ga_s, patm) / lv

    # Cap at available water
    soil_evap = jnp.minimum(wat_sto, erate)

    # PORT-BRANCH: water.ground_evaporation.snow_floor_zero
    # Condition: SWE > eps -> no evaporation from floor if snow on ground
    soil_evap = jnp.where(swe > eps, 0.0, soil_evap)

    return soil_evap


def canopy_water_snow(
    state: CanopyWaterState,
    params: CanopySnowParams,
    tc: jax.Array,
    pre: jax.Array,
    ae: jax.Array,
    d: jax.Array,
    ra: jax.Array,
    u: jax.Array,
    lai: jax.Array,
    patm: jax.Array,
    time_step: jax.Array,
) -> tuple[CanopyWaterState, CanopySnowFlux]:
    """Canopy interception, throughfall and snowpack dynamics.

    Fortran: canopy_water_snow in water_mod.f90
    Handles precipitation partitioning (rain/snow), canopy interception,
    sublimation/evaporation, snow unloading, melt/freeze, and liquid drainage.

    Args:
        state: Initial canopy water state
        params: Canopy snow parameters
        tc: Air temperature (°C)
        pre: Precipitation rate (mm/s)
        ae: Available energy (W/m²)
        d: Vapor pressure deficit (Pa)
        ra: Canopy aerodynamic resistance (s/m)
        u: Wind speed (m/s)
        lai: Leaf area index (m²/m²)
        patm: Atmospheric pressure (Pa)
        time_step: Timestep (hours)

    Returns:
        (new_state, flux) — updated CanopyWaterState and CanopySnowFlux
    """
    eps = 1e-16
    tmin = 0.0   # below: all snow
    tmax = 1.0   # above: all rain
    tmelt = 0.0  # melt threshold

    dt = time_step * 3600.0

    # PORT-BRANCH: water.canopy_water_snow.precip_phase
    # Condition: T<=Tmin -> all snow; T>=Tmax -> all rain; between -> mixed
    fw = jnp.clip((tc - tmin) / (tmax - tmin), 0.0, 1.0)
    fs = 1.0 - fw

    # Canopy storage capacities (mm)
    wmax_tot = params.wmax * lai
    wmaxsnow_tot = params.wmaxsnow * lai

    # Latent heats (J/kg)
    lv = 1.0e3 * (3147.5 - 2.37 * (tc + 273.15))
    ls = lv + 3.3e5

    # Accumulated precipitation (mm during timestep)
    prec = pre * dt

    # Initial state
    wo = state.CanopyStorage
    sweo = state.SWE
    w = wo
    swe_i = state.swe_i
    swe_l = state.swe_l

    # Aerodynamic conductance
    ga = 1.0 / ra

    # ── Evaporation/sublimation from canopy ──
    # Guard: use max(w, eps) in Ce to avoid 0^(-0.4)=Inf which produces
    # NaN via Inf*0 in gi.  Fortran avoids this through if/else short-
    # circuit evaluation; JAX evaluates all branches unconditionally.
    w_safe = jnp.maximum(w, eps)
    ce = 0.01 * (w_safe / jnp.maximum(wmaxsnow_tot, eps)) ** (-0.4)  # exposure coeff
    sh = 1.79 + 3.0 * jnp.sqrt(u)  # Sherwood number
    gi = sh * w_safe * ce / 7.68 + eps   # conductance for sublimation (m/s)

    # Sublimation rate (when no precip and T <= Tmin)
    erate_sublim = dt / ls * penman_monteith(ae, d, tc, gi, ga, patm)
    # Evaporation rate (when no precip and T > Tmin)
    gs_free = 1.0e6
    erate_evap = dt / lv * penman_monteith(ae, d, tc, gs_free, ga, patm)

    # PORT-BRANCH: water.canopy_water_snow.sublim_vs_evap
    # Condition: Prec==0 & T<=Tmin -> sublimation; Prec==0 & T>Tmin -> evaporation
    no_precip = prec <= 0.0
    erate = jnp.where(
        no_precip & (tc <= tmin), erate_sublim,
        jnp.where(no_precip & (tc > tmin), erate_evap, 0.0),
    )
    # PORT-BRANCH: water.canopy_water_snow.lai_evap_guard
    # Condition: LAI <= eps -> no canopy evaporation/sublimation
    erate = jnp.where(lai > eps, erate, 0.0)

    # PORT-BRANCH: water.canopy_water_snow.snow_unloading
    # Condition: T >= Tmin -> unload excess beyond wmax_tot
    unload = jnp.where(tc >= tmin, jnp.maximum(w - wmax_tot, 0.0), 0.0)
    w = w - unload

    # PORT-BRANCH: water.canopy_water_snow.interception_phase
    # Condition: T < Tmin -> snow interception; else -> liquid interception
    # Guard division by zero with jnp.maximum
    interc_snow = jnp.where(
        lai > eps,
        (wmaxsnow_tot - w) * (1.0 - jnp.exp(-prec / jnp.maximum(wmaxsnow_tot, eps))),
        0.0,
    )
    interc_rain = jnp.where(
        lai > eps,
        jnp.maximum(0.0, wmax_tot - w) * (1.0 - jnp.exp(-prec / jnp.maximum(wmax_tot, eps))),
        0.0,
    )
    interc = jnp.where(tc < tmin, interc_snow, interc_rain)

    # Update canopy storage after interception
    w = w + interc

    # Evaporate from canopy (clamp w >= 0 — canopy storage is non-negative)
    canopy_evap = jnp.minimum(erate, jnp.maximum(0.0, w))
    w = jnp.maximum(0.0, w - canopy_evap)

    # Throughfall
    trfall = prec + unload - interc

    # PORT-BRANCH: water.canopy_water_snow.melt_freeze
    # Condition: T>=Tmelt -> melt; T<Tmelt & swe_l>0 -> freeze; else -> nothing
    melt = jnp.where(
        tc >= tmelt,
        jnp.minimum(swe_i, params.kmelt * dt * (tc - tmelt)),
        0.0,
    )
    freeze = jnp.where(
        (tc < tmelt) & (swe_l > 0.0),
        jnp.minimum(swe_l, params.kfreeze * dt * (tmelt - tc)),
        0.0,
    )

    # Snow ice and liquid
    sice = jnp.maximum(0.0, swe_i + fs * trfall + freeze - melt)
    sliq = jnp.maximum(0.0, swe_l + fw * trfall - freeze + melt)

    # Potential infiltration — excess liquid drains from snowpack
    pot_infil = jnp.maximum(0.0, sliq - sice * params.frac_snowliq)
    sliq = jnp.maximum(0.0, sliq - pot_infil)

    # New state
    new_state = CanopyWaterState(
        CanopyStorage=w,
        SWE=sliq + sice,
        swe_i=sice,
        swe_l=sliq,
    )

    # Mass-balance error
    mbe = (new_state.CanopyStorage + new_state.SWE) - (wo + sweo) - (prec - canopy_evap - pot_infil)

    flux = CanopySnowFlux(
        Throughfall=trfall,
        Interception=interc,
        CanopyEvap=canopy_evap,
        Unloading=unload,
        PotInfiltration=pot_infil,
        Melt=melt,
        Freeze=freeze,
        mbe=mbe,
    )

    return new_state, flux


def canopy_water_flux(
    rn: jax.Array,
    ta: jax.Array,
    prec: jax.Array,
    vpd: jax.Array,
    u: jax.Array,
    patm: jax.Array,
    fapar: jax.Array,
    lai: jax.Array,
    cw_state: CanopyWaterState,
    sw_beta: jax.Array,
    sw_wat_sto: jax.Array,
    aero_params: AeroParams,
    cs_params: CanopySnowParams,
    time_step: jax.Array,
) -> tuple[CanopyWaterState, CanopyWaterFlux]:
    """Canopy water balance orchestrator.

    Fortran: canopy_water_flux in water_mod.f90
    Calls aerodynamics → canopy_water_snow → ground_evaporation.

    Args:
        rn: Net radiation (W/m²)
        ta: Air temperature (°C)
        prec: Precipitation rate (mm/s)
        vpd: Vapor pressure deficit (Pa)
        u: Wind speed (m/s)
        patm: Pressure (Pa)
        fapar: Fraction absorbed PAR (-)
        lai: Leaf area index
        cw_state: Initial canopy water state
        sw_beta: Soil evaporation modifier (-)
        sw_wat_sto: Soil water storage (mm)
        aero_params: Aerodynamic parameters
        cs_params: Canopy snow parameters
        time_step: Timestep (hours)

    Returns:
        (new_cw_state, flux) — updated state and full CanopyWaterFlux
    """
    # Aerodynamic resistances
    ra, _rb, ras, _ustar, _uh, _ug = aerodynamics(lai, u, aero_params)

    # Canopy interception & snowpack
    ae_canopy = rn * fapar
    new_state, cs_flux = canopy_water_snow(
        cw_state, cs_params, ta, prec, ae_canopy, vpd, ra, u, lai, patm, time_step,
    )

    # Ground evaporation
    ae_soil = rn * (1.0 - fapar)
    soil_evap = ground_evaporation(
        ta, ae_soil, vpd, ras, patm, new_state.SWE, sw_beta, sw_wat_sto,
        cs_params.gsoil, time_step,
    )

    flux = CanopyWaterFlux(
        Throughfall=cs_flux.Throughfall,
        Interception=cs_flux.Interception,
        CanopyEvap=cs_flux.CanopyEvap,
        Unloading=cs_flux.Unloading,
        SoilEvap=soil_evap,
        ET=jnp.array(0.0),
        Transpiration=jnp.array(0.0),
        PotInfiltration=cs_flux.PotInfiltration,
        Melt=cs_flux.Melt,
        Freeze=cs_flux.Freeze,
        mbe=cs_flux.mbe,
    )

    return new_state, flux


def soil_water(
    state: SoilWaterState,
    soil_params: SoilHydroParams,
    max_poros: jax.Array,
    potinf: jax.Array,
    tr: jax.Array,
    evap: jax.Array,
    latflow: jax.Array,
    time_step: jax.Array,
) -> tuple[SoilWaterState, SoilWaterFlux, jax.Array, jax.Array, jax.Array]:
    """Soil water balance in 1-layer bucket.

    Fortran: soil_water in water_mod.f90

    Args:
        state: Initial soil water state
        soil_params: Van Genuchten hydraulic parameters
        max_poros: Maximum porosity (m³/m³)
        potinf: Potential infiltration (mm)
        tr: Transpiration (mm) — may be clamped to available water
        evap: Evaporation (mm) — may be clamped to available water
        latflow: Lateral flow input (mm) — reset to 0 internally
        time_step: Timestep (hours)

    Returns:
        (new_state, flux, tr_out, evap_out, latflow_out)
    """
    eps = 1e-16
    dt_s = time_step * 3600.0

    wat_sto_0 = state.WatSto
    pond_sto_0 = state.PondSto

    # Potential infiltration includes existing pond storage
    rr = potinf + pond_sto_0

    # Clamp transpiration and evaporation to available water
    tr_out = jnp.minimum(tr, wat_sto_0 + rr - eps)
    evap_out = jnp.minimum(evap, wat_sto_0 + rr - tr_out - eps)

    # Water storage after upward fluxes
    wat_sto = wat_sto_0 - tr_out - evap_out

    # Vertical drainage (limited to surplus above field capacity)
    drain = state.Kh * (dt_s * 1000.0)  # mm
    drain = jnp.minimum(drain, jnp.maximum(0.0, wat_sto - state.FcSto))

    # Lateral drainage (hardcoded to 0 in Fortran)
    latflow_out = jnp.array(0.0)
    latflow_out = jnp.maximum(0.0, jnp.minimum(latflow_out, wat_sto - drain - eps))

    # Infiltration limited by available water or storage space
    infil = jnp.minimum(rr, state.MaxWatSto + drain + latflow_out)

    # Update soil water storage
    wat_sto = wat_sto + infil - drain - latflow_out

    # Pond storage and runoff (excess water)
    to_pond = jnp.maximum(0.0, rr - infil)
    pond_sto = jnp.minimum(to_pond, state.MaxPondSto)
    runoff = jnp.maximum(0.0, to_pond - pond_sto)

    # Derived state: volumetric water content
    wliq = max_poros * jnp.minimum(1.0, wat_sto / state.MaxWatSto)
    sat = wliq / max_poros
    beta = jnp.minimum(1.0, wliq / max_poros)

    # Soil water potential and hydraulic conductivity
    psi = soil_water_retention_curve(wliq, soil_params)
    kh = soil_hydraulic_conductivity(wliq, soil_params)

    # Mass-balance error
    mbe = (wat_sto - wat_sto_0) + (pond_sto - pond_sto_0) \
        - (rr - tr_out - evap_out - drain - latflow_out - runoff)

    new_state = SoilWaterState(
        WatSto=wat_sto,
        PondSto=pond_sto,
        MaxWatSto=state.MaxWatSto,
        MaxPondSto=state.MaxPondSto,
        FcSto=state.FcSto,
        Wliq=wliq,
        Psi=psi,
        Sat=sat,
        Kh=kh,
        beta=beta,
    )

    flux = SoilWaterFlux(
        Infiltration=infil,
        Runoff=runoff,
        Drainage=drain,
        LateralFlow=latflow_out,
        ET=tr_out + evap_out,
        mbe=mbe,
    )

    return new_state, flux, tr_out, evap_out, latflow_out
