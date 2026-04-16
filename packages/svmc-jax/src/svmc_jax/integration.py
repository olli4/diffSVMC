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
    hourly_driver: jax.Array   # (24, 7) stacked [temp, rg, prec, vpd, pres, co2, wind]
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


class IntegrationForcing(NamedTuple):
    """Public forcing container for grouped and vmappable integration calls."""
    hourly_driver: jax.Array     # (n_days, 24, 7)
    daily_lai: jax.Array         # (n_days,)
    daily_manage_type: jax.Array   # (n_days,)
    daily_manage_c_in: jax.Array   # (n_days,)
    daily_manage_c_out: jax.Array  # (n_days,)


class PhydroRunParams(NamedTuple):
    """Grouped P-Hydro parameters for integration runs."""
    conductivity: float
    psi50: float
    b_param: float
    alpha_cost: float
    gamma_cost: float
    rdark: float


class WaterRunParams(NamedTuple):
    """Grouped SpaFHy canopy, aero, and soil parameters."""
    soil_depth: float
    max_poros: float
    fc: float
    wp: float
    ksat: float
    n_van: float
    watres: float
    alpha_van: float
    watsat: float
    maxpond: float
    wmax: float
    wmaxsnow: float
    kmelt: float
    kfreeze: float
    frac_snowliq: float
    gsoil: float
    hc: float
    w_leaf: float
    rw: float
    rwmin: float
    zmeas: float
    zground: float
    zo_ground: float


class AllocationRunParams(NamedTuple):
    """Grouped allocation parameters for integration runs."""
    cratio_resp: float
    cratio_leaf: float
    cratio_root: float
    cratio_biomass: float
    harvest_index: float
    turnover_cleaf: float
    turnover_croot: float
    sla: float
    q10: float
    invert_option: int


class YassoInitParams(NamedTuple):
    """Grouped Yasso initialization parameters for integration runs."""
    yasso_param: jax.Array
    yasso_totc: float
    yasso_cn_input: float
    yasso_fract_root: float
    yasso_fract_legacy: float
    yasso_tempr_c: float
    yasso_precip_day: float
    yasso_tempr_ampl: float


class IntegrationParams(NamedTuple):
    """Grouped static integration parameters suitable for JAX pytrees."""
    phydro: PhydroRunParams
    water: WaterRunParams
    allocation: AllocationRunParams
    yasso: YassoInitParams
    pft_is_oat: float


# ── Initialization ────────────────────────────────────────────────────


def _to_daily_forcing(forcing: IntegrationForcing) -> DailyForcing:
    return DailyForcing(
        hourly_driver=forcing.hourly_driver,
        lai=forcing.daily_lai,
        management_type=forcing.daily_manage_type,
        management_c_in=forcing.daily_manage_c_in,
        management_c_out=forcing.daily_manage_c_out,
    )


def _stack_hourly_driver(
    hourly_temp: jax.Array,
    hourly_rg: jax.Array,
    hourly_prec: jax.Array,
    hourly_vpd: jax.Array,
    hourly_pres: jax.Array,
    hourly_co2: jax.Array,
    hourly_wind: jax.Array,
) -> jax.Array:
    return jnp.stack([
        hourly_temp,
        hourly_rg,
        hourly_prec,
        hourly_vpd,
        hourly_pres,
        hourly_co2,
        hourly_wind,
    ], axis=-1)


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
    phydro_optimizer: str,
):
    """Return a scan-compatible hourly step function closed over static params."""

    time_step_arr = jnp.asarray(_TIME_STEP)
    smoothing_alpha = jnp.asarray([_ALPHA_SMOOTH1, _ALPHA_SMOOTH2])
    smoothing_keep = 1.0 - smoothing_alpha
    rdark_arr = jnp.float64(rdark)
    max_poros_arr = jnp.asarray(max_poros)
    latflow = jnp.asarray(0.0)
    false_scalar = jnp.asarray(False)

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
            rdark_leaf=rdark_arr,
            conductivity=conductivity,
            psi50=psi50,
            b_param=b_param,
            alpha=alpha,
            gamma_cost=gamma_cost,
            solver_kind=phydro_optimizer,
        )

        aj = phydro_result.aj
        gs = phydro_result.gs
        vcmax_hr = phydro_result.vcmax

        # GPP (T/ha equivalent units: mol/m²/s → T C /ha)
        # gpp_hr = (aj + rdark * vcmax_hr) * c_molmass * 1e-6 * 1e-3 * lai
        gpp_raw = (aj + rdark_arr * vcmax_hr) * _C_MOLMASS * 1.0e-6 * 1.0e-3 * lai
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
        new_sw_state, _sw_flux, _tr_out, _evap_out, _latflow_out = soil_water(
            carry.sw_state, soil_params, max_poros_arr,
            cw_flux.PotInfiltration, tr_spafhy, cw_flux.SoilEvap,
            latflow, time_step_arr,
        )

        # ── Exponential met smoothing ──
        met_temp = tc   # °C
        met_prec = prec + cw_flux.Melt / (_TIME_STEP * 3600.0)
        met_daily = jnp.stack((met_temp, met_prec))

        # JAX-compatible exponential smoothing
        smoothed = smoothing_alpha * met_daily + smoothing_keep * carry.met_rolling
        new_met_rolling = jnp.where(carry.is_first_met, met_daily, smoothed)
        new_is_first_met = false_scalar

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
    phydro_optimizer: str,
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

    zero = jnp.asarray(0.0)
    unit_year = jnp.asarray(1.0)
    cratio_resp_arr = jnp.asarray(cratio_resp)
    cratio_leaf_arr = jnp.asarray(cratio_leaf)
    cratio_root_arr = jnp.asarray(cratio_root)
    cratio_biomass_arr = jnp.asarray(cratio_biomass)
    harvest_index_arr = jnp.asarray(harvest_index)
    turnover_cleaf_arr = jnp.asarray(turnover_cleaf)
    turnover_croot_arr = jnp.asarray(turnover_croot)
    sla_arr = jnp.asarray(sla)
    q10_arr = jnp.asarray(q10)
    invert_option_arr = jnp.float64(invert_option)
    pft_is_oat_arr = jnp.asarray(pft_is_oat)

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
            phydro_optimizer=phydro_optimizer,
        )

        # Initialize hourly carry with zeroed accumulators
        hourly_init = HourlyCarry(
            cw_state=carry.cw_state,
            sw_state=carry.sw_state,
            met_rolling=carry.met_rolling,
            is_first_met=carry.is_first_met,
            temp_acc=zero,
            precip_acc=zero,
            gpp_acc=zero,
            vcmax_acc=zero,
            num_gpp=zero,
            num_vcmax=zero,
            et_acc=zero,
        )

        # Run 24 hourly steps
        final_hourly, _ = jax.lax.scan(hourly_step, hourly_init, forcing.hourly_driver)

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
            cratio_resp_arr, cratio_leaf_arr,
            cratio_root_arr, cratio_biomass_arr,
            harvest_index_arr,
            turnover_cleaf_arr, turnover_croot_arr,
            sla_arr, q10_arr, invert_option_arr,
            forcing.management_type, forcing.management_c_in, forcing.management_c_out,
            pft_is_oat_arr, carry.pheno,
        )
        # invert_alloc modifies cratio_leaf, cratio_root, turnover_cleaf, litter_cleaf, cleaf
        inv_cratio_leaf = inv_result.cratio_leaf
        inv_cratio_root = inv_result.cratio_root
        inv_turnover_cleaf = inv_result.turnover_cleaf
        inv_litter_cleaf = inv_result.litter_cleaf
        inv_cleaf = inv_result.cleaf

        # ── Allocation: alloc_hypothesis_2 ──
        alloc_result = alloc_hypothesis_2(
            temp_avg, gpp_avg, leaf_rdark_day,
            carry.croot, inv_cleaf, carry.cstem, carry.cgrain,
            inv_litter_cleaf, carry.grain_fill,
            cratio_resp_arr, inv_cratio_leaf,
            inv_cratio_root, cratio_biomass_arr,
            inv_turnover_cleaf, turnover_croot_arr,
            sla_arr, q10_arr, invert_option_arr,
            forcing.management_type, forcing.management_c_in, forcing.management_c_out,
            pft_is_oat_arr, carry.pheno,
        )

        new_cleaf = alloc_result.cleaf
        new_croot = alloc_result.croot
        new_cstem = alloc_result.cstem
        new_cgrain = alloc_result.cgrain
        new_litter_cleaf = alloc_result.litter_cleaf
        new_litter_croot = alloc_result.litter_croot
        new_compost = alloc_result.compost
        new_lai_alloc = alloc_result.lai
        new_above = alloc_result.abovebiomass
        new_below = alloc_result.belowbiomass
        new_yield = alloc_result.yield_
        new_grain_fill = alloc_result.grain_fill
        new_pheno = jnp.float64(alloc_result.pheno_stage)
        auto_resp_day = alloc_result.auto_resp
        npp_day = alloc_result.npp_day

        # ── Yasso decomposition ──
        input_cfract = inputs_to_fractions(
            new_litter_cleaf, new_litter_croot,
            carry.soluble, new_compost,
        )
        ctend, ntend = decompose(
            yasso_param, unit_year, temp_avg, precip_acc,
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


def run_integration_grouped(
    forcing: IntegrationForcing,
    params: IntegrationParams,
    phydro_optimizer: str = "projected_lbfgs",
) -> tuple[DailyCarry, DailyOutput]:
    """Run the full SVMC integration loop with grouped, vmappable inputs."""
    from .yasso.initialize_totc import initialize_totc

    soil_params = SoilHydroParams(
        watsat=jnp.asarray(params.water.watsat),
        watres=jnp.asarray(params.water.watres),
        alpha_van=jnp.asarray(params.water.alpha_van),
        n_van=jnp.asarray(params.water.n_van),
        ksat=jnp.asarray(params.water.ksat),
    )
    aero_params = AeroParams(
        hc=jnp.asarray(params.water.hc),
        zmeas=jnp.asarray(params.water.zmeas),
        zground=jnp.asarray(params.water.zground),
        zo_ground=jnp.asarray(params.water.zo_ground),
        w_leaf=jnp.asarray(params.water.w_leaf),
    )
    cs_params = CanopySnowParams(
        wmax=jnp.asarray(params.water.wmax),
        wmaxsnow=jnp.asarray(params.water.wmaxsnow),
        kmelt=jnp.asarray(params.water.kmelt),
        kfreeze=jnp.asarray(params.water.kfreeze),
        frac_snowliq=jnp.asarray(params.water.frac_snowliq),
        gsoil=jnp.asarray(params.water.gsoil),
    )

    cw_state, sw_state = initialization_spafhy(
        params.water.soil_depth,
        params.water.max_poros,
        params.water.fc,
        params.water.maxpond,
        soil_params,
    )

    cstate, nstate = initialize_totc(
        params.yasso.yasso_param,
        jnp.asarray(params.yasso.yasso_totc),
        jnp.asarray(params.yasso.yasso_cn_input),
        jnp.asarray(params.yasso.yasso_fract_root),
        jnp.asarray(params.yasso.yasso_fract_legacy),
        jnp.asarray(params.yasso.yasso_tempr_c),
        jnp.asarray(params.yasso.yasso_precip_day),
        jnp.asarray(params.yasso.yasso_tempr_ampl),
    )

    init_carry = DailyCarry(
        cw_state=cw_state,
        sw_state=sw_state,
        met_rolling=jnp.zeros(2),
        is_first_met=jnp.asarray(True),
        cleaf=jnp.asarray(0.0),
        croot=jnp.asarray(0.0),
        cstem=jnp.asarray(0.0),
        cgrain=jnp.asarray(0.0),
        litter_cleaf=jnp.asarray(0.0),
        litter_croot=jnp.asarray(0.0),
        compost=jnp.asarray(0.0),
        soluble=jnp.asarray(0.0),
        above=jnp.asarray(0.0),
        below=jnp.asarray(0.0),
        yield_=jnp.asarray(0.0),
        grain_fill=jnp.asarray(0.0),
        lai_alloc=jnp.asarray(0.0),
        pheno=jnp.asarray(1.0),
        cstate=cstate,
        nstate=nstate,
        lai_prev=jnp.asarray(0.0),
    )

    daily_step = _make_daily_step(
        aero_params=aero_params,
        cs_params=cs_params,
        soil_params=soil_params,
        max_poros=params.water.max_poros,
        rdark=params.phydro.rdark,
        conductivity=params.phydro.conductivity,
        psi50=params.phydro.psi50,
        b_param=params.phydro.b_param,
        alpha=params.phydro.alpha_cost,
        gamma_cost=params.phydro.gamma_cost,
        phydro_optimizer=phydro_optimizer,
        cratio_resp=params.allocation.cratio_resp,
        cratio_leaf=params.allocation.cratio_leaf,
        cratio_root=params.allocation.cratio_root,
        cratio_biomass=params.allocation.cratio_biomass,
        harvest_index=params.allocation.harvest_index,
        turnover_cleaf=params.allocation.turnover_cleaf,
        turnover_croot=params.allocation.turnover_croot,
        sla=params.allocation.sla,
        q10=params.allocation.q10,
        invert_option=params.allocation.invert_option,
        yasso_param=params.yasso.yasso_param,
        pft_is_oat=params.pft_is_oat,
    )

    final_carry, daily_outputs = jax.lax.scan(
        daily_step,
        init_carry,
        _to_daily_forcing(forcing),
    )
    return final_carry, daily_outputs


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
    phydro_optimizer: str = "projected_lbfgs",
) -> tuple[DailyCarry, DailyOutput]:
    """Run the full SVMC integration loop.

    Returns:
        (final_carry, daily_outputs) where daily_outputs has per-day
        trajectories for comparison against the integration fixture.
    """
    forcing = IntegrationForcing(
        hourly_driver=_stack_hourly_driver(
            hourly_temp,
            hourly_rg,
            hourly_prec,
            hourly_vpd,
            hourly_pres,
            hourly_co2,
            hourly_wind,
        ),
        daily_lai=daily_lai,
        daily_manage_type=daily_manage_type,
        daily_manage_c_in=daily_manage_c_in,
        daily_manage_c_out=daily_manage_c_out,
    )
    params = IntegrationParams(
        phydro=PhydroRunParams(
            conductivity=conductivity,
            psi50=psi50,
            b_param=b_param,
            alpha_cost=alpha_cost,
            gamma_cost=gamma_cost,
            rdark=rdark,
        ),
        water=WaterRunParams(
            soil_depth=soil_depth,
            max_poros=max_poros,
            fc=fc,
            wp=wp,
            ksat=ksat,
            n_van=n_van,
            watres=watres,
            alpha_van=alpha_van,
            watsat=watsat,
            maxpond=maxpond,
            wmax=wmax,
            wmaxsnow=wmaxsnow,
            kmelt=kmelt,
            kfreeze=kfreeze,
            frac_snowliq=frac_snowliq,
            gsoil=gsoil,
            hc=hc,
            w_leaf=w_leaf,
            rw=rw,
            rwmin=rwmin,
            zmeas=zmeas,
            zground=zground,
            zo_ground=zo_ground,
        ),
        allocation=AllocationRunParams(
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
        ),
        yasso=YassoInitParams(
            yasso_param=yasso_param,
            yasso_totc=yasso_totc,
            yasso_cn_input=yasso_cn_input,
            yasso_fract_root=yasso_fract_root,
            yasso_fract_legacy=yasso_fract_legacy,
            yasso_tempr_c=yasso_tempr_c,
            yasso_precip_day=yasso_precip_day,
            yasso_tempr_ampl=yasso_tempr_ampl,
        ),
        pft_is_oat=pft_is_oat,
    )
    return run_integration_grouped(
        forcing,
        params,
        phydro_optimizer=phydro_optimizer,
    )
