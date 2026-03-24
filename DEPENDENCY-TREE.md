# SVMC Submodel Dependency Tree

The SVMC vegetation process model (Soil-Vegetation Model Coupled) couples:
- **P-Hydro**: photosynthesis–hydraulics optimization (Joshi et al. 2022)
- **SpaFHy**: canopy & soil water balance (Launiainen et al. 2019)
- **Carbon Allocation**: biomass partitioning & phenology
- **YASSO20**: soil organic carbon decomposition

## Time scales

| Loop     | Submodels called                                                  |
| -------- | ----------------------------------------------------------------- |
| Hourly   | read climate → P-Hydro → canopy_water_flux → soil_water          |
| Daily    | allocation (invert_alloc, alloc_hypothesis_2) → YASSO decompose → NEE |
| Yearly   | YASSO annual (mod5c20) spin-up (optional)                         |

## Full dependency tree

Leaf-level functions (no submodel deps) are at the bottom; the main program is
at the top. Indentation shows "calls / depends on".

```
SVMC main loop
├── readctrl_namelist                          [config]
├── readvegpara_namelist                       [config]
├── readsoilhydro_namelist                     [config]
├── readsoilyasso_namelist                     [config]
├── readalloc_namelist                         [config]
├── initialization_spafhy                      [init]
│   ├── soil_water_retention_curve             ← LEAF
│   └── soil_hydraulic_conductivity            ← LEAF
├── wrapper_yasso_initialize_totc              [init]
│   └── yasso.initialize_totc                  ← LEAF (Yasso20 core)
│
├── [hourly loop]
│   ├── netCDF_readClim (+ readlai, readsoilmoist, readsnow, readmanagement)
│   ├── soil_water_retention_curve             ← LEAF (if obs_soilmoist)
│   │
│   ├── pmodel_hydraulics_numerical            [P-HYDRO]
│   │   ├── calc_kmm
│   │   │   └── ftemp_arrh                     ← LEAF
│   │   ├── gammastar
│   │   │   └── ftemp_arrh                     ← LEAF
│   │   ├── ftemp_kphio                        ← LEAF
│   │   ├── viscosity_h2o
│   │   │   └── density_h2o                    ← LEAF
│   │   ├── density_h2o                        ← LEAF
│   │   ├── optimise_midterm_multi (L-BFGS-B)
│   │   │   └── fn_profit
│   │   │       ├── calc_gs
│   │   │       │   └── scale_conductivity     ← LEAF
│   │   │       └── calc_assim_light_limited
│   │   │           └── quadratic              ← LEAF
│   │   ├── calc_gs                            (reused)
│   │   └── calc_assim_light_limited           (reused)
│   │
│   ├── canopy_water_flux                      [SPAFHY-CANOPY]
│   │   ├── aerodynamics                       ← LEAF
│   │   ├── canopy_water_snow
│   │   │   └── penman_monteith
│   │   │       └── e_sat                      ← LEAF
│   │   └── ground_evaporation
│   │       └── penman_monteith                (reused)
│   │
│   ├── soil_water                             [SPAFHY-SOIL]
│   │   ├── soil_water_retention_curve         ← LEAF
│   │   └── soil_hydraulic_conductivity        ← LEAF
│   │
│   └── exponential_smooth_met                 ← LEAF
│
├── [daily loop]
│   ├── invert_alloc                           [ALLOCATION]
│   ├── alloc_hypothesis_2                     [ALLOCATION]
│   ├── inputs_to_fractions                    ← LEAF (litter → AWENH)
│   ├── wrapper_yasso_decompose                [YASSO-DAILY]
│   │   └── yasso.decompose                    ← Yasso20 core
│   └── NEE = TotalResp − GPP
│
└── [yearly, optional]
    └── wrapper_yasso_annual                   [YASSO-ANNUAL]
        └── yasso20.mod5c20                    ← Yasso20 core
```

## Porting order (leaves first)

We port bottom-up so each submodel can be tested against reference data before
being composed into higher-level models.

### Phase 1 — Leaf functions (no dependencies)

| # | Function                       | Inputs                           | Outputs              |
|---|--------------------------------|----------------------------------|----------------------|
| 1 | `ftemp_arrh`                   | tk, dha                          | scaling factor       |
| 2 | `gammastar`                    | tc, patm                         | Γ* (Pa)              |
| 3 | `ftemp_kphio`                  | tc, c4                           | φ₀ scaling           |
| 4 | `density_h2o`                  | tc, patm                         | ρ (kg/m³)            |
| 5 | `viscosity_h2o`                | tc, patm                         | μ (Pa·s)             |
| 6 | `e_sat`                        | T, P                             | esat, s, γ           |
| 7 | `penman_monteith`              | AE, D, T, Gs, Ga, P             | LE (W/m²)            |
| 8 | `quadratic`                    | a, b, c                          | root r1              |
| 9 | `soil_water_retention_curve`   | vol_liq, params                  | ψ (MPa)              |
| 10| `soil_hydraulic_conductivity`  | vol_liq, params                  | K (m/s)              |
| 11| `aerodynamics`                 | LAI, Uo, params                  | ra, rb, ras, u*, …   |
| 12| `scale_conductivity`           | K, par_env                       | K (mol/m²/s/MPa)     |
| 13| `exponential_smooth_met`       | met_daily, met_rolling, ind      | met_rolling          |
| 14| `inputs_to_fractions`          | leaf_c, root_c, sol, comp        | AWENH fractions      |

### Phase 2 — Mid-level functions

| # | Function                       | Depends on (from Phase 1)               |
|---|--------------------------------|-----------------------------------------|
| 1 | `calc_kmm`                     | ftemp_arrh                              |
| 2 | `calc_gs`                      | scale_conductivity                      |
| 3 | `calc_assim_light_limited`     | quadratic                               |
| 4 | `fn_profit`                    | calc_gs, calc_assim_light_limited       |

### Phase 3 — High-level submodels

| # | Submodel                         | Depends on                              |
|---|----------------------------------|-----------------------------------------|
| 1 | `canopy_water_snow`              | penman_monteith                         |
| 2 | `ground_evaporation`             | penman_monteith                         |
| 3 | `canopy_water_flux`              | aerodynamics, canopy_water_snow, ground_evaporation |
| 4 | `soil_water`                     | soil_water_retention_curve, soil_hydraulic_conductivity |
| 5 | `optimise_midterm_multi`         | fn_profit (L-BFGS-B optimizer)          |
| 6 | `pmodel_hydraulics_numerical`    | calc_kmm, gammastar, ftemp_kphio, viscosity_h2o, density_h2o, optimise_midterm_multi |
| 7 | `alloc_hypothesis_2` / `invert_alloc` | standalone (uses GPP, leaf dark resp.) |
| 8 | `wrapper_yasso_decompose`        | Yasso20 core (decompose)                |
| 9 | `wrapper_yasso_annual`           | Yasso20 core (mod5c20)                  |

### Phase 4 — Full SVMC integration

Compose all submodels into the hourly + daily + yearly loop.

## Key interfaces for logging

To generate reference data from the Fortran model, add logging at these call
boundaries:

1. **P-Hydro inputs/outputs**: before/after `pmodel_hydraulics_numerical`
2. **Canopy water inputs/outputs**: before/after `canopy_water_flux`
3. **Soil water inputs/outputs**: before/after `soil_water`
4. **Allocation inputs/outputs**: before/after `invert_alloc` + `alloc_hypothesis_2`
5. **YASSO inputs/outputs**: before/after `wrapper_yasso_decompose`
6. **Leaf functions**: individual function calls (for unit testing)

## Notes on differentiability

- The L-BFGS-B optimizer in `optimise_midterm_multi` uses finite-difference
  gradients. In JAX, replace with `jax.grad` through `fn_profit` directly.
- The `quadratic` solver needs a differentiable formulation (handle the
  discriminant edge case smoothly).
- YASSO20's `mod5c20` is a matrix exponential; naturally differentiable.
- All `min`/`max` clamps need smooth replacements (e.g. softmin/softmax) or
  `jax.lax.cond` for clean autodiff.
