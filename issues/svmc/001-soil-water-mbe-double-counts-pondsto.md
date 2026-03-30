# soil_water mass-balance error double-counts PondSto

**File:** `src/water_mod.f90`, subroutine `soil_water`  
**Severity:** Low (diagnostic-only; does not affect state evolution)

## Summary

The mass-balance error (mbe) diagnostic in `soil_water` systematically reports a
nonzero residual whenever pond storage is nonzero, even though the actual water
balance is conserved.  The cause is that `PondSto_old` enters the flux
bookkeeping via `rr` but is not cancelled out in the mbe formula.

## Reproducer

Any call to `soil_water` where `soilwater_state%PondSto > 0` at entry will
produce `mbe ≠ 0`.

## Analysis

At line 128, available water for infiltration is computed as:

```fortran
rr = potinf + soilwater_state%PondSto          ! line 128
PondSto = 0.0                                   ! line 129
```

At lines 185–187, the mass-balance error is:

```fortran
soilwater_flux%mbe = (WatSto_new - WatSto0)              &   ! ΔWatSto
                   + (PondSto_new - PondSto0)             &   ! ΔPondSto
                   - (rr - tr - evap - drain - latflow - runoff)  ! net flux
```

Expanding `rr = potinf + PondSto0`:

```
mbe = ΔWatSto + ΔPondSto - (potinf + PondSto0 - tr - evap - drain - latflow - runoff)
```

The `PondSto0` term inside `rr` does not cancel with anything in the storage
deltas, introducing a systematic bias of `−PondSto0`.

A correct formulation would either:

1. Use `rr = potinf` (exclude PondSto from rr) and account for pond-to-soil
   transfer as a separate internal flux, **or**
2. Adjust the mbe formula to subtract `potinf` instead of `rr`:

```fortran
soilwater_flux%mbe = (WatSto_new - WatSto0)           &
                   + (PondSto_new - PondSto0)          &
                   - (potinf - tr - evap - drain - latflow - runoff)
```

## Impact

- The mbe field is a diagnostic only; `WatSto` and `PondSto` themselves evolve correctly.
- However, the unreliable mbe silently masks any real conservation violations that
  might arise from future code changes.

## Provenance

Found during a systematic differentiable port of SVMC to JAX and TypeScript.
Verified present in upstream `huitang-earth/SVMC` main branch (no local
modifications to `water_mod.f90`).
