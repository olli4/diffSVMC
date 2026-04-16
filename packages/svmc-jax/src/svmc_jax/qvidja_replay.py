"""Shared helpers for Qvidja reference replay inputs."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

_YASSO_PARAM = (
    0.51, 5.19, 0.13, 0.1, 0.5, 0.0, 1.0, 1.0, 0.99, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.163, 0.0, -0.0, 0.0, 0.0, 0.0,
    0.0, 0.158, -0.002, 0.17, -0.005, 0.067, -0.0, -1.44,
    -2.0, -6.9, 0.0042, 0.0015, -2.55, 1.24, 0.25,
)


def build_qvidja_run_inputs(ref: Mapping[str, Any], ndays: int):
    """Build grouped forcing and parameter inputs for run_integration_grouped."""
    from .integration import (
        AllocationRunParams,
        IntegrationForcing,
        IntegrationParams,
        PhydroRunParams,
        WaterRunParams,
        YassoInitParams,
    )

    defaults = ref["defaults"]
    hourly = ref["hourly"]
    daily = ref["daily"]
    nhours = ndays * 24

    forcing = IntegrationForcing(
        hourly_driver=jnp.stack([
            jnp.array(hourly["temp_hr"][:nhours]).reshape(ndays, 24),
            jnp.array(hourly["rg_hr"][:nhours]).reshape(ndays, 24),
            jnp.array(hourly["prec_hr"][:nhours]).reshape(ndays, 24),
            jnp.array(hourly["vpd_hr"][:nhours]).reshape(ndays, 24),
            jnp.array(hourly["pres_hr"][:nhours]).reshape(ndays, 24),
            jnp.array(hourly["co2_hr"][:nhours]).reshape(ndays, 24),
            jnp.array(hourly["wind_hr"][:nhours]).reshape(ndays, 24),
        ], axis=-1),
        daily_lai=jnp.array(daily["lai_day"][:ndays]),
        daily_manage_type=jnp.array([float(x) for x in daily["manage_type"][:ndays]]),
        daily_manage_c_in=jnp.array(daily["manage_c_in"][:ndays]),
        daily_manage_c_out=jnp.array(daily["manage_c_out"][:ndays]),
    )
    params = IntegrationParams(
        phydro=PhydroRunParams(
            conductivity=defaults["conductivity"],
            psi50=defaults["psi50"],
            b_param=defaults["b"],
            alpha_cost=defaults["alpha"],
            gamma_cost=defaults["gamma"],
            rdark=defaults["rdark"],
        ),
        water=WaterRunParams(
            soil_depth=defaults["soil_depth"],
            max_poros=defaults["max_poros"],
            fc=defaults["fc"],
            wp=defaults["wp"],
            ksat=defaults["ksat"],
            n_van=1.14,
            watres=0.0,
            alpha_van=5.92,
            watsat=defaults["max_poros"],
            maxpond=0.0,
            wmax=0.5,
            wmaxsnow=4.5,
            kmelt=2.8934e-5,
            kfreeze=5.79e-6,
            frac_snowliq=0.05,
            gsoil=5.0e-3,
            hc=0.6,
            w_leaf=0.01,
            rw=0.20,
            rwmin=0.02,
            zmeas=2.0,
            zground=0.1,
            zo_ground=0.01,
        ),
        allocation=AllocationRunParams(
            cratio_resp=defaults["cratio_resp"],
            cratio_leaf=defaults["cratio_leaf"],
            cratio_root=defaults["cratio_root"],
            cratio_biomass=defaults["cratio_biomass"],
            harvest_index=defaults["harvest_index"],
            turnover_cleaf=defaults["turnover_cleaf"],
            turnover_croot=defaults["turnover_croot"],
            sla=defaults["sla"],
            q10=defaults["q10"],
            invert_option=defaults["invert_option"],
        ),
        yasso=YassoInitParams(
            yasso_param=jnp.array(_YASSO_PARAM),
            yasso_totc=defaults["yasso_totc"],
            yasso_cn_input=defaults["yasso_cn_input"],
            yasso_fract_root=defaults["yasso_fract_root"],
            yasso_fract_legacy=0.0,
            yasso_tempr_c=5.4,
            yasso_precip_day=1.87,
            yasso_tempr_ampl=20.0,
        ),
        pft_is_oat=0.0,
    )
    return forcing, params


def build_qvidja_run_kwargs(ref: Mapping[str, Any], ndays: int) -> dict[str, Any]:
    """Build the shared run_integration kwargs for the Qvidja replay."""
    forcing, params = build_qvidja_run_inputs(ref, ndays)
    return {
        "hourly_temp": forcing.hourly_driver[..., 0],
        "hourly_rg": forcing.hourly_driver[..., 1],
        "hourly_prec": forcing.hourly_driver[..., 2],
        "hourly_vpd": forcing.hourly_driver[..., 3],
        "hourly_pres": forcing.hourly_driver[..., 4],
        "hourly_co2": forcing.hourly_driver[..., 5],
        "hourly_wind": forcing.hourly_driver[..., 6],
        "daily_lai": forcing.daily_lai,
        "daily_manage_type": forcing.daily_manage_type,
        "daily_manage_c_in": forcing.daily_manage_c_in,
        "daily_manage_c_out": forcing.daily_manage_c_out,
        "conductivity": params.phydro.conductivity,
        "psi50": params.phydro.psi50,
        "b_param": params.phydro.b_param,
        "alpha_cost": params.phydro.alpha_cost,
        "gamma_cost": params.phydro.gamma_cost,
        "rdark": params.phydro.rdark,
        "soil_depth": params.water.soil_depth,
        "max_poros": params.water.max_poros,
        "fc": params.water.fc,
        "wp": params.water.wp,
        "ksat": params.water.ksat,
        "n_van": params.water.n_van,
        "watres": params.water.watres,
        "alpha_van": params.water.alpha_van,
        "watsat": params.water.watsat,
        "maxpond": params.water.maxpond,
        "wmax": params.water.wmax,
        "wmaxsnow": params.water.wmaxsnow,
        "kmelt": params.water.kmelt,
        "kfreeze": params.water.kfreeze,
        "frac_snowliq": params.water.frac_snowliq,
        "gsoil": params.water.gsoil,
        "hc": params.water.hc,
        "w_leaf": params.water.w_leaf,
        "rw": params.water.rw,
        "rwmin": params.water.rwmin,
        "zmeas": params.water.zmeas,
        "zground": params.water.zground,
        "zo_ground": params.water.zo_ground,
        "cratio_resp": params.allocation.cratio_resp,
        "cratio_leaf": params.allocation.cratio_leaf,
        "cratio_root": params.allocation.cratio_root,
        "cratio_biomass": params.allocation.cratio_biomass,
        "harvest_index": params.allocation.harvest_index,
        "turnover_cleaf": params.allocation.turnover_cleaf,
        "turnover_croot": params.allocation.turnover_croot,
        "sla": params.allocation.sla,
        "q10": params.allocation.q10,
        "invert_option": params.allocation.invert_option,
        "pft_is_oat": params.pft_is_oat,
        "yasso_param": params.yasso.yasso_param,
        "yasso_totc": params.yasso.yasso_totc,
        "yasso_cn_input": params.yasso.yasso_cn_input,
        "yasso_fract_root": params.yasso.yasso_fract_root,
        "yasso_fract_legacy": params.yasso.yasso_fract_legacy,
        "yasso_tempr_c": params.yasso.yasso_tempr_c,
        "yasso_precip_day": params.yasso.yasso_precip_day,
        "yasso_tempr_ampl": params.yasso.yasso_tempr_ampl,
    }