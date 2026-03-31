# soil_water: implicit SAVE on `runoff` and `to_pond` produces stale flux outputs

**File:** `src/water_mod.f90`, subroutine `soil_water`  
**Severity:** Low (diagnostic-only; does not affect state evolution)

## Summary

The local variables `runoff` and `to_pond` are declared with initializers
(`real(8) :: ... to_pond=0.0, runoff=0.0`), giving them the implicit Fortran
SAVE attribute. Both are only assigned inside the `if (infil < rr)` guard
(line 155), so on calls where all available water infiltrates (`infil >= rr`),
they retain stale values from the most recent call that did produce ponding.

## Mechanism

```fortran
! Declaration (line 114):              ~~~ implicit SAVE ~~~
real(8) :: drain=0.0, infil=0.0, to_pond=0.0, runoff=0.0

! ...

! Conditional assignment (lines 155-160):
if (infil < rr) then
  to_pond = rr - infil
  PondSto = min(PondSto + to_pond, soilwater_state%MaxPondSto)
  runoff = max(0.0, to_pond - PondSto)
end if

! Unconditional flux output (line 167):
soilwater_flux%Runoff = runoff     ! ← stale when infil >= rr

! Unconditional MBE (line 186-188):
soilwater_flux%mbe = ... - (rr - tr - evap - drain - latflow - runoff)
!                                                              ^^^^^^
!                                              stale runoff corrupts MBE
```

When `infil >= rr`:
- `to_pond` should be 0 but retains its last nonzero value (harmless — only
  read inside the if-block where it's assigned)
- `runoff` should be 0 but retains its last nonzero value → leaks into
  `soilwater_flux%Runoff` and the MBE calculation

## Impact

- **State evolution is unaffected.** `WatSto` and `PondSto` are computed
  independently of `runoff`. The stale value only corrupts the reported
  `soilwater_flux%Runoff` output and the MBE diagnostic.
- This compounds with the PondSto double-counting bug documented in
  [001-soil-water-mbe-double-counts-pondsto.md](001-soil-water-mbe-double-counts-pondsto.md),
  making the MBE diagnostic doubly unreliable.
- `soilwater_flux%Runoff` is consumed only for CSV/netCDF output in
  `SVMC.f90`, not fed back into state evolution.

## Also safe: `drain` and `infil`

`drain` and `infil` also have initializers on the same line but are
unconditionally assigned before use (lines 144 and 150 respectively), so
their SAVE attribute is harmless.

## Fix

Remove declaration initializers and add explicit initialization:

```fortran
! Replace:
real(8) :: drain=0.0, infil=0.0, to_pond=0.0, runoff=0.0

! With:
real(8) :: drain, infil, to_pond, runoff
drain = 0.0
infil = 0.0
to_pond = 0.0
runoff = 0.0
```

## JAX implementation

The JAX port handles this correctly — Python locals are always fresh each
call. The `runoff` and `to_pond` equivalents are computed with `jnp.where`
and default to 0.0 when no ponding occurs.
