"""Phase 5: Full SVMC integration loop (hourly + daily).

Composes P-Hydro, canopy/soil water balance, allocation, and Yasso
decomposition into a complete daily time-stepping model suitable for
``jax.lax.scan``.

The public entry point is :func:`run_integration`, which takes
initialization parameters, forcing data, and Yasso20 MAP parameters,
returning per-day state trajectories for validation against the
integration fixture.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from .phydro.solver import pmodel_hydraulics_numerical
from .phydro.leaf_functions import density_h2o
from .water.canopy_soil import (
    CanopyWaterState,
    CanopySnowParams,
    CanopyWaterFlux,
    SoilWaterState,
    canopy_water_flux,
    soil_water,
)
from .water.leaf_functions import (
    AeroParams,
    SoilHydroParams,
    soil_water_retention_curve,
    soil_hydraulic_conductivity,
)
from .allocation.alloc_hypothesis_2 import alloc_hypothesis_2
from .allocation.invert_alloc import invert_alloc
from .yasso.leaf_functions import inputs_to_fractions
from .yasso.decompose import decompose


# ── Constants ─────────────────────────────────────────────────────────

_C_MOLMASS = 12.0107       # g/mol — carbon molecular mass
_H2O_MOLMASS = 18.01528    # g/mol — water molecular mass
_K_EXT = 0.5               # Beer's law extinction coefficient
_TIME_STEP = 1.0           # hours per hourly step

# Exponential smoothing coefficients (from wrapper_yasso.f90)
_ALPHA_SMOOTH1 = 0.01      # temperature smoothing
_ALPHA_SMOOTH2 = 0.0016    # precipitation smoothing

# LAI guard threshold (same as Fortran harness)
_LAI_GUARD = 1.0e-6


# ── Typed containers for scan carry states ────────────────────────────


class HourlyCarry(NamedTuple):
    """State carried across hourly steps within one day."""
    cw_state: CanopyWaterState       # canopy water state
    sw_state: SoilWaterState         # soil water state
    met_rolling: jax.Array           # (2,) exponentially smoothed [temp_C, precip]
    is_first_met: jax.Array          # scalar bool — first met smoothing call
    temp_acc: jax.Array              # accumulated smoothed temperature
    precip_acc: jax.Array            # accumulated smoothed precipitation (mm)
    gpp_acc: jax.Array               # accumulated GPP (T/ha/hr equiv)
    vcmax_acc: jax.Array             # accumulated Vcmax (µmol/m²/s)
    num_gpp: jax.Array               # count of valid GPP hours
    num_vcmax: jax.Array             # count of valid Vcmax hours
    et_acc: jax.Array                # accumulated ET (mm)


class DailyCarry(NamedTuple):
    """State carried across daily steps."""
    cw_state: CanopyWaterState
    sw_state: SoilWaterState
    met_rolling: jax.Array           # (2,) smoothed met
    is_first_met: jax.Array          # bool
    # Carbon pools
    cleaf: jax.Array
    croot: jax.Array
    cstem: jax.Array
    cgrain: jax.Array
    litter_cleaf: jax.Array
    litter_croot: jax.Array
    compost: jax.Array
    soluble: jax.Array
    above: jax.Array
    below: jax.Array
    yield_: jax.Array
    grain_fill: jax.Array
    lai_alloc: jax.Array
    pheno: jax.Array
    # Yasso state
    cstate: jax.Array                # (5,) AWENH carbon pools
    nstate: jax.Array                # scalar nitrogen state
    # Previous day LAI (for delta_lai)
    lai_prev: jax.Array


class DailyForcing(NamedTuple):
    """Per-day forcing data for the outer scan."""
    hourly_temp: jax.Array     # (24,) temperature (K)
    hourly_rg: jax.Array       # (24,) global radiation (W/m²)
    hourly_prec: jax.Array     # (24,) precipitation (kg/m²/s)
    hourly_vpd: jax.Array      # (24,) VPD (Pa)
    hourly_pres: jax.Array     # (24,) pressure (Pa)
    hourly_co2: jax.Array      # (24,) CO2 (mol/mol)
    hourly_wind: jax.Array     # (24,) wind speed (m/s)
    lai: jax.Array             # scalar — observed LAI for this day
    management_type: jax.Array # scalar — management type (0-4)
    management_c_in: jax.Array # scalar
    management_c_out: jax.Array  # scalar


class DailyOutput(NamedTuple):
    """Per-day outputs for validation against the integration fixture."""
    gpp_avg: jax.Array
    nee: jax.Array
    hetero_resp: jax.Array
    auto_resp: jax.Array
    cleaf: jax.Array
    croot: jax.Array
    cstem: jax.Array
    cgrain: jax.Array
    lai_alloc: jax.Array
    litter_cleaf: jax.Array
    litter_croot: jax.Array
    soc_total: jax.Array
    wliq: jax.Array
    psi: jax.Array
    cstate: jax.Array          # (5,) AWENH pools
    et_total: jax.Array        # daily accumulated ET (mm)


# ── Initialization ────────────────────────────────────────────────────


def initialization_spafhy(
    soil_depth: float,
    max_poros: float,
    fc: float,
    maxpond: float,
    soil_params: SoilHydroParams,
) -> tuple[CanopyWaterState, SoilWaterState]:
    """Initialize canopy and soil water states (90% saturation cold start).

    Replicates Fortran ``initialization_spafhy`` in water_mod.f90.
    """
    max_wat_sto = 1000.0 * soil_depth * max_poros
    fc_sto = 1000.0 * soil_depth * fc
    max_pond_sto = maxpond

    wat_sto = 0.9 * max_wat_sto
    pond_sto = 0.0
    wliq = max_poros * jnp.minimum(1.0, wat_sto / max_wat_sto)
    sat = wliq / max_poros
    beta = jnp.minimum(1.0, wliq / max_poros)
    psi = soil_water_retention_curve(wliq, soil_params)
    kh = soil_hydraulic_conductivity(wliq, soil_params)

    cw_state = CanopyWaterState(
        CanopyStorage=jnp.array(0.0),
        SWE=jnp.array(0.0),
        swe_i=jnp.array(0.0),
        swe_l=jnp.array(0.0),
    )
    sw_state = SoilWaterState(
        WatSto=jnp.asarray(wat_sto),
        PondSto=jnp.asarray(pond_sto),
        MaxWatSto=jnp.asarray(max_wat_sto),
        MaxPondSto=jnp.asarray(max_pond_sto),
        FcSto=jnp.asarray(fc_sto),
        Wliq=jnp.asarray(wliq),
        Psi=jnp.asarray(psi),
        Sat=jnp.asarray(sat),
        Kh=jnp.asarray(kh),
        beta=jnp.asarray(beta),
    )
    return cw_state, sw_state


# ── Hourly step ───────────────────────────────────────────────────────


def _make_hourly_step(
    lai: jax.Array,
    fapar: jax.Array,
    aero_params: AeroParams,
    cs_params: CanopySnowParams,
    soil_params: SoilHydroParams,
    max_poros: float,
    rdark: float,
    conductivity: float,
    psi50: float,
    b_param: float,
    alpha: float,
    gamma_cost: float,
):
    """Return a scan-compatible hourly step function closed over static params."""

    time_step_arr = jnp.array(_TIME_STEP)

    def hourly_step(carry: HourlyCarry, forcing: jax.Array) -> tuple[HourlyCarry, None]:
        """One hourly time step: P-Hydro → water balance → met smoothing → accumulate.

        Args:
            carry: HourlyCarry state
            forcing: (7,) array [temp_K, rg, prec, vpd, pres, co2, wind]
        """
        temp_k = forcing[0]
        rg = forcing[1]
        prec = forcing[2]
        vpd = forcing[3]
        pres = forcing[4]
        co2 = forcing[5]
        wind = forcing[6]
        tc = temp_k - 273.15

        # ── P-Hydro ──
        psi_soil = carry.sw_state.Psi
        # PPFD: rg * 2.1 / LAI (same as Fortran harness)
        ppfd = rg * 2.1 / jnp.maximum(lai, _LAI_GUARD)
        co2_ppm = co2 * 1.0e6

        phydro_result = pmodel_hydraulics_numerical(
            tc, ppfd, vpd, co2_ppm, pres, fapar, psi_soil,
            rdark_leaf=jnp.array(rdark),
            conductivity=conductivity,
            psi50=psi50,
            b_param=b_param,
            alpha=alpha,
            gamma_cost=gamma_cost,
        )

        aj = phydro_result["aj"]
        gs = phydro_result["gs"]
        vcmax_hr = phydro_result["vcmax"]

        # GPP (T/ha equivalent units: mol/m²/s → T C /ha)
        # gpp_hr = (aj + rdark * vcmax_hr) * c_molmass * 1e-6 * 1e-3 * lai
        gpp_raw = (aj + rdark * vcmax_hr) * _C_MOLMASS * 1.0e-6 * 1.0e-3 * lai
        # Guard: zero outputs when LAI is negligible or there is no light.
        # Without light the P-Hydro optimizer solves a degenerate problem
        # (Iabs = 0 → the quadratic for aj is analytically zero).  Tiny
        # floating-point noise in that quadratic can flip the sign of aj
        # nondeterministically, corrupting the num_gpp averaging
        # denominator.  Matching the Fortran harness guard:
        #   if (int_lai > 1.0d-6 .and. int_rg > 0.0d0)
        has_light = (lai > _LAI_GUARD) & (rg > 0.0)
        gpp_hr = jnp.where(has_light, gpp_raw, 0.0)
        aj_guarded = jnp.where(has_light, aj, 0.0)
        gs_guarded = jnp.where(has_light, gs, 0.0)
        vcmax_guarded = jnp.where(has_light, vcmax_hr, 0.0)

        # ── Transpiration ──
        # tr_phydro = 1.6 * gs * (vpd/pres) * Mw/density * lai  (m/s)
        rho_w = density_h2o(tc, pres)
        tr_raw = 1.6 * gs_guarded * (vpd / pres) * _H2O_MOLMASS / rho_w * lai
        gs_is_finite = jnp.isfinite(gs_guarded)
        tr_phydro = jnp.where(gs_is_finite & has_light, tr_raw, 0.0)

        # ── Net radiation ──
        rn = rg * 0.7

        # ── Canopy water flux ──
        new_cw_state, cw_flux = canopy_water_flux(
            rn, tc, prec, vpd, wind, pres, fapar, lai,
            carry.cw_state, carry.sw_state.beta, carry.sw_state.WatSto,
            aero_params, cs_params, time_step_arr,
        )

        # Update ET with P-Hydro transpiration
        tr_spafhy = tr_phydro * (_TIME_STEP * 3600.0)
        et_hr = tr_spafhy + cw_flux.SoilEvap + cw_flux.CanopyEvap

        # ── Soil water ──
        latflow = jnp.array(0.0)
        new_sw_state, _sw_flux, _tr_out, _evap_out, _latflow_out = soil_water(
            carry.sw_state, soil_params, jnp.array(max_poros),
            cw_flux.PotInfiltration, tr_spafhy, cw_flux.SoilEvap,
            latflow, time_step_arr,
        )

        # ── Exponential met smoothing ──
        met_temp = tc   # °C
        met_prec = prec + cw_flux.Melt / (_TIME_STEP * 3600.0)
        met_daily = jnp.array([met_temp, met_prec])

        # JAX-compatible exponential smoothing
        smoothed = jnp.array([
            _ALPHA_SMOOTH1 * met_daily[0] + (1.0 - _ALPHA_SMOOTH1) * carry.met_rolling[0],
            _ALPHA_SMOOTH2 * met_daily[1] + (1.0 - _ALPHA_SMOOTH2) * carry.met_rolling[1],
        ])
        new_met_rolling = jnp.where(carry.is_first_met, met_daily, smoothed)
        new_is_first_met = jnp.array(False)

        # ── Accumulate ──
        # GPP valid check: aj >= 0 and not NaN
        gpp_valid = jnp.isfinite(aj_guarded) & (aj_guarded >= 0.0)
        gpp_to_add = jnp.where(gpp_valid, gpp_hr, 0.0)
        new_num_gpp = carry.num_gpp + jnp.where(gpp_valid, 1.0, 0.0)

        # Vcmax valid check
        vcmax_valid = jnp.isfinite(vcmax_guarded) & (vcmax_guarded > 0.0)
        vcmax_to_add = jnp.where(vcmax_valid, vcmax_guarded, 0.0)
        new_num_vcmax = carry.num_vcmax + jnp.where(vcmax_valid, 1.0, 0.0)

        new_carry = HourlyCarry(
            cw_state=new_cw_state,
            sw_state=new_sw_state,
            met_rolling=new_met_rolling,
            is_first_met=new_is_first_met,
            temp_acc=carry.temp_acc + new_met_rolling[0],
            precip_acc=carry.precip_acc + new_met_rolling[1] * _TIME_STEP * 3600.0,
            gpp_acc=carry.gpp_acc + gpp_to_add,
            vcmax_acc=carry.vcmax_acc + vcmax_to_add,
            num_gpp=new_num_gpp,
            num_vcmax=new_num_vcmax,
            et_acc=carry.et_acc + et_hr,
        )
        return new_carry, None

    return hourly_step


# ── Daily step ────────────────────────────────────────────────────────


def _make_daily_step(
    aero_params: AeroParams,
    cs_params: CanopySnowParams,
    soil_params: SoilHydroParams,
    max_poros: float,
    rdark: float,
    conductivity: float,
    psi50: float,
    b_param: float,
    alpha: float,
    gamma_cost: float,
    # Allocation parameters (fixed)
    cratio_resp: float,
    cratio_leaf: float,
    cratio_root: float,
    cratio_biomass: float,
    harvest_index: float,
    turnover_cleaf: float,
    turnover_croot: float,
    sla: float,
    q10: float,
    invert_option: int,
    # Yasso parameters
    yasso_param: jax.Array,
    # PFT flag
    pft_is_oat: float,
):
    """Return a scan-compatible daily step function."""

    def daily_step(carry: DailyCarry, forcing: DailyForcing) -> tuple[DailyCarry, DailyOutput]:
        lai = forcing.lai
        delta_lai = lai - carry.lai_prev
        fapar = 1.0 - jnp.exp(-_K_EXT * lai)

        # Build hourly step for this day's LAI
        hourly_step = _make_hourly_step(
            lai=lai, fapar=fapar,
            aero_params=aero_params, cs_params=cs_params,
            soil_params=soil_params, max_poros=max_poros,
            rdark=rdark, conductivity=conductivity,
            psi50=psi50, b_param=b_param,
            alpha=alpha, gamma_cost=gamma_cost,
        )

        # Stack hourly forcing: shape (24, 7)
        hourly_forcing = jnp.stack([
            forcing.hourly_temp, forcing.hourly_rg, forcing.hourly_prec,
            forcing.hourly_vpd, forcing.hourly_pres, forcing.hourly_co2,
            forcing.hourly_wind,
        ], axis=-1)

        # Initialize hourly carry with zeroed accumulators
        hourly_init = HourlyCarry(
            cw_state=carry.cw_state,
            sw_state=carry.sw_state,
            met_rolling=carry.met_rolling,
            is_first_met=carry.is_first_met,
            temp_acc=jnp.array(0.0),
            precip_acc=jnp.array(0.0),
            gpp_acc=jnp.array(0.0),
            vcmax_acc=jnp.array(0.0),
            num_gpp=jnp.array(0.0),
            num_vcmax=jnp.array(0.0),
            et_acc=jnp.array(0.0),
        )

        # Run 24 hourly steps
        final_hourly, _ = jax.lax.scan(hourly_step, hourly_init, hourly_forcing)

        # ── Daily averages ──
        temp_avg = final_hourly.temp_acc / 24.0
        gpp_avg = jnp.where(
            final_hourly.num_gpp > 0,
            final_hourly.gpp_acc / final_hourly.num_gpp,
            0.0,
        )
        vcmax_avg = jnp.where(
            final_hourly.num_vcmax > 0,
            final_hourly.vcmax_acc / final_hourly.num_vcmax,
            0.0,
        )
        precip_acc = final_hourly.precip_acc
        leaf_rdark_day = rdark * vcmax_avg * _C_MOLMASS * 1.0e-6 * 1.0e-3 * lai

        # ── Allocation: invert_alloc ──
        inv_result = invert_alloc(
            jnp.asarray(delta_lai), leaf_rdark_day, temp_avg, gpp_avg,
            carry.litter_cleaf, carry.cleaf, carry.cstem,
            jnp.array(cratio_resp), jnp.array(cratio_leaf),
            jnp.array(cratio_root), jnp.array(cratio_biomass),
            jnp.array(harvest_index),
            jnp.array(turnover_cleaf), jnp.array(turnover_croot),
            jnp.array(sla), jnp.array(q10), jnp.array(float(invert_option)),
            forcing.management_type, forcing.management_c_in, forcing.management_c_out,
            jnp.array(pft_is_oat), carry.pheno,
        )
        # invert_alloc modifies cratio_leaf, cratio_root, turnover_cleaf, litter_cleaf, cleaf
        inv_cratio_leaf = inv_result["cratio_leaf"]
        inv_cratio_root = inv_result["cratio_root"]
        inv_turnover_cleaf = inv_result["turnover_cleaf"]
        inv_litter_cleaf = inv_result["litter_cleaf"]
        inv_cleaf = inv_result["cleaf"]

        # ── Allocation: alloc_hypothesis_2 ──
        alloc_result = alloc_hypothesis_2(
            temp_avg, gpp_avg, leaf_rdark_day,
            carry.croot, inv_cleaf, carry.cstem, carry.cgrain,
            inv_litter_cleaf, carry.grain_fill,
            jnp.array(cratio_resp), inv_cratio_leaf,
            inv_cratio_root, jnp.array(cratio_biomass),
            inv_turnover_cleaf, jnp.array(turnover_croot),
            jnp.array(sla), jnp.array(q10), jnp.array(float(invert_option)),
            forcing.management_type, forcing.management_c_in, forcing.management_c_out,
            jnp.array(pft_is_oat), carry.pheno,
        )

        new_cleaf = alloc_result["cleaf"]
        new_croot = alloc_result["croot"]
        new_cstem = alloc_result["cstem"]
        new_cgrain = alloc_result["cgrain"]
        new_litter_cleaf = alloc_result["litter_cleaf"]
        new_litter_croot = alloc_result["litter_croot"]
        new_compost = alloc_result["compost"]
        new_lai_alloc = alloc_result["lai"]
        new_above = alloc_result["abovebiomass"]
        new_below = alloc_result["belowbiomass"]
        new_yield = alloc_result["yield"]
        new_grain_fill = alloc_result["grain_fill"]
        new_pheno = jnp.float64(alloc_result["pheno_stage"])
        auto_resp_day = alloc_result["auto_resp"]
        npp_day = alloc_result["npp_day"]

        # ── Yasso decomposition ──
        input_cfract = inputs_to_fractions(
            new_litter_cleaf, new_litter_croot,
            carry.soluble, new_compost,
        )
        ctend, ntend = decompose(
            yasso_param, jnp.array(1.0), temp_avg, precip_acc,
            carry.cstate, carry.nstate,
        )
        new_cstate = carry.cstate + ctend + input_cfract
        new_nstate = carry.nstate + ntend

        # ── Respiration & NEE ──
        hetero_resp = jnp.sum(-ctend) / 24.0 / 3600.0
        auto_resp_flux = auto_resp_day / 24.0 / 3600.0
        total_resp = hetero_resp + auto_resp_flux
        nee = total_resp - gpp_avg

        # ── Build new carry ──
        new_carry = DailyCarry(
            cw_state=final_hourly.cw_state,
            sw_state=final_hourly.sw_state,
            met_rolling=final_hourly.met_rolling,
            is_first_met=final_hourly.is_first_met,
            cleaf=new_cleaf,
            croot=new_croot,
            cstem=new_cstem,
            cgrain=new_cgrain,
            litter_cleaf=new_litter_cleaf,
            litter_croot=new_litter_croot,
            compost=new_compost,
            soluble=carry.soluble,
            above=new_above,
            below=new_below,
            yield_=new_yield,
            grain_fill=new_grain_fill,
            lai_alloc=new_lai_alloc,
            pheno=new_pheno,
            cstate=new_cstate,
            nstate=new_nstate,
            lai_prev=lai,
        )

        output = DailyOutput(
            gpp_avg=gpp_avg,
            nee=nee,
            hetero_resp=hetero_resp,
            auto_resp=auto_resp_flux,
            cleaf=new_cleaf,
            croot=new_croot,
            cstem=new_cstem,
            cgrain=new_cgrain,
            lai_alloc=new_lai_alloc,
            litter_cleaf=new_litter_cleaf,
            litter_croot=new_litter_croot,
            soc_total=jnp.sum(new_cstate),
            wliq=final_hourly.sw_state.Wliq,
            psi=final_hourly.sw_state.Psi,
            cstate=new_cstate,
            et_total=final_hourly.et_acc,
        )

        return new_carry, output

    return daily_step


# ── Public entry point ────────────────────────────────────────────────


def run_integration(
    # Forcing data
    hourly_temp: jax.Array,     # (n_days, 24)
    hourly_rg: jax.Array,       # (n_days, 24)
    hourly_prec: jax.Array,     # (n_days, 24)
    hourly_vpd: jax.Array,      # (n_days, 24)
    hourly_pres: jax.Array,     # (n_days, 24)
    hourly_co2: jax.Array,      # (n_days, 24)
    hourly_wind: jax.Array,     # (n_days, 24)
    daily_lai: jax.Array,       # (n_days,)
    daily_manage_type: jax.Array,   # (n_days,)
    daily_manage_c_in: jax.Array,   # (n_days,)
    daily_manage_c_out: jax.Array,  # (n_days,)
    # P-Hydro parameters
    conductivity: float,
    psi50: float,
    b_param: float,
    alpha_cost: float,
    gamma_cost: float,
    rdark: float,
    # SpaFHy soil parameters
    soil_depth: float,
    max_poros: float,
    fc: float,
    wp: float,
    ksat: float,
    n_van: float,
    watres: float,
    alpha_van: float,
    watsat: float,
    maxpond: float,
    # SpaFHy canopy/aero parameters
    wmax: float,
    wmaxsnow: float,
    kmelt: float,
    kfreeze: float,
    frac_snowliq: float,
    gsoil: float,
    hc: float,
    w_leaf: float,
    rw: float,
    rwmin: float,
    zmeas: float,
    zground: float,
    zo_ground: float,
    # Allocation parameters
    cratio_resp: float,
    cratio_leaf: float,
    cratio_root: float,
    cratio_biomass: float,
    harvest_index: float,
    turnover_cleaf: float,
    turnover_croot: float,
    sla: float,
    q10: float,
    invert_option: int,
    pft_is_oat: float,
    # Yasso initialization
    yasso_param: jax.Array,     # (35,)
    yasso_totc: float,
    yasso_cn_input: float,
    yasso_fract_root: float,
    yasso_fract_legacy: float,
    yasso_tempr_c: float,
    yasso_precip_day: float,
    yasso_tempr_ampl: float,
) -> tuple[DailyCarry, DailyOutput]:
    """Run the full SVMC integration loop.

    Returns:
        (final_carry, daily_outputs) where daily_outputs has per-day
        trajectories for comparison against the integration fixture.
    """
    from .yasso.initialize_totc import initialize_totc

    # ── Build parameter structures ──
    soil_params = SoilHydroParams(
        watsat=jnp.array(watsat),
        watres=jnp.array(watres),
        alpha_van=jnp.array(alpha_van),
        n_van=jnp.array(n_van),
        ksat=jnp.array(ksat),
    )
    aero_params = AeroParams(
        hc=jnp.array(hc),
        zmeas=jnp.array(zmeas),
        zground=jnp.array(zground),
        zo_ground=jnp.array(zo_ground),
        w_leaf=jnp.array(w_leaf),
    )
    cs_params = CanopySnowParams(
        wmax=jnp.array(wmax),
        wmaxsnow=jnp.array(wmaxsnow),
        kmelt=jnp.array(kmelt),
        kfreeze=jnp.array(kfreeze),
        frac_snowliq=jnp.array(frac_snowliq),
        gsoil=jnp.array(gsoil),
    )

    # ── Initialize states ──
    cw_state, sw_state = initialization_spafhy(
        soil_depth, max_poros, fc, maxpond, soil_params,
    )

    # Yasso soil carbon state
    cstate, nstate = initialize_totc(
        yasso_param, jnp.array(yasso_totc), jnp.array(yasso_cn_input),
        jnp.array(yasso_fract_root), jnp.array(yasso_fract_legacy),
        jnp.array(yasso_tempr_c), jnp.array(yasso_precip_day),
        jnp.array(yasso_tempr_ampl),
    )

    # ── Initial daily carry ──
    init_carry = DailyCarry(
        cw_state=cw_state,
        sw_state=sw_state,
        met_rolling=jnp.zeros(2),
        is_first_met=jnp.array(True),
        cleaf=jnp.array(0.0),
        croot=jnp.array(0.0),
        cstem=jnp.array(0.0),
        cgrain=jnp.array(0.0),
        litter_cleaf=jnp.array(0.0),
        litter_croot=jnp.array(0.0),
        compost=jnp.array(0.0),
        soluble=jnp.array(0.0),
        above=jnp.array(0.0),
        below=jnp.array(0.0),
        yield_=jnp.array(0.0),
        grain_fill=jnp.array(0.0),
        lai_alloc=jnp.array(0.0),
        pheno=jnp.array(1.0),
        cstate=cstate,
        nstate=nstate,
        lai_prev=jnp.array(0.0),
    )

    # ── Build forcing structure ──
    forcing = DailyForcing(
        hourly_temp=hourly_temp,
        hourly_rg=hourly_rg,
        hourly_prec=hourly_prec,
        hourly_vpd=hourly_vpd,
        hourly_pres=hourly_pres,
        hourly_co2=hourly_co2,
        hourly_wind=hourly_wind,
        lai=daily_lai,
        management_type=daily_manage_type,
        management_c_in=daily_manage_c_in,
        management_c_out=daily_manage_c_out,
    )

    # ── Build daily step and run ──
    daily_step = _make_daily_step(
        aero_params=aero_params,
        cs_params=cs_params,
        soil_params=soil_params,
        max_poros=max_poros,
        rdark=rdark,
        conductivity=conductivity,
        psi50=psi50,
        b_param=b_param,
        alpha=alpha_cost,
        gamma_cost=gamma_cost,
        cratio_resp=cratio_resp,
        cratio_leaf=cratio_leaf,
        cratio_root=cratio_root,
        cratio_biomass=cratio_biomass,
        harvest_index=harvest_index,
        turnover_cleaf=turnover_cleaf,
        turnover_croot=turnover_croot,
        sla=sla,
        q10=q10,
        invert_option=invert_option,
        yasso_param=yasso_param,
        pft_is_oat=pft_is_oat,
    )

    final_carry, daily_outputs = jax.lax.scan(daily_step, init_carry, forcing)
    return final_carry, daily_outputs
