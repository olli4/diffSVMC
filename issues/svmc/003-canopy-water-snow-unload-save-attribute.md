# canopy_water_snow: Fortran SAVE attribute bugs on `Unload` and `Interc`

**File:** `src/water_mod.f90`, subroutine `canopy_water_snow`  
**Severity:** High (affects state evolution — creates phantom snow mass)

## Summary

The local variable `Unload` is declared with an initializer
(`real(8) :: Unload=0.0`), which in Fortran implicitly gives it the SAVE
attribute. Because `Unload` is only assigned inside the `if (T >= Tmin)` guard
(line 430), every call with `T < Tmin` reuses the stale value from the most
recent warm call. This stale `Unload` enters the throughfall calculation
(`Trfall = Prec + Unload - Interc`, line 462) and adds phantom snow to the
snowpack every cold hour.

## Mechanism

```
Declaration (line 357):               ~~~ implicit SAVE ~~~
   real(8) :: Unload=0.0, Interc=0.0, Melt=0.0, Freeze=0.0, Trfall=0.0

Warm call (T >= Tmin):
   Unload = max(W - wmax_tot, 0.0)    → e.g. 0.0764 mm
   W = W - Unload                     → canopy is reduced

Cold call (T < Tmin):
   Unload is NOT assigned             → retains 0.0764 from warm call
   Trfall = Prec + Unload - Interc    → 0 + 0.0764 − 0 = 0.0764
   Sice += fS * Trfall                → adds 0.0764 mm phantom snow
   BUT: W = W - Unload is NOT executed (inside the T>=Tmin guard)
   → Water appears from nowhere
```

The per-call MBE diagnostic (line 499-501) cannot detect the bug because it
treats the stale `Unload` as a real flux — the throughfall "enters" the
snowpack consistent with the MBE formula within a single call. The accounting
identity holds within each call; it just can't know `Unload` is a ghost from a
prior invocation.

## Evidence

Using the Qvidja site data and hourly-level tracing (Jan 10–14, 2022):

| Metric | Fortran | JAX |
|--------|---------|-----|
| SWE growth per cold hour (zero precip, T<0) | **+0.0764 mm** (constant) | **+0.0000 mm** |
| SWE difference after 5-day cold period | +11.39 mm | — |
| Soil moisture spike at thaw (Jan 14-17) | +0.0118 (excess meltwater) | — |

Key observations:
- **41 consecutive hours** show bit-for-bit identical SWE growth of
  `+0.076386378359` mm — impossible with varying temperature/radiation.
- Growth is exactly constant because the same stale `Unload` is replayed.
- Growth stops immediately when `T >= Tmin` (Unload is recomputed fresh).
- All other fluxes (CanopyEvap, Interc, Melt, Freeze) match between Fortran
  and JAX to machine precision.
- The accumulated phantom SWE (11.39mm) melts during the Jan 14-17 thaw event,
  producing excess meltwater that raises soil moisture by ~0.012 (wliq).

## Affected variables

Only `Unload` and `Interc` cause bugs. The other initialized locals (`Melt`,
`Freeze`, `Trfall`, `CanopyEvap`, `PotInfil`, `Sice`, `Sliq`) are all
unconditionally assigned before use in every code path, so their SAVE
attribute is harmless.

### Bug 1: `Unload` (HIGH)

Already described above — stale warm-weather unloading leaks into every cold
hour, creating phantom snow mass.

### Bug 2: `Interc` (MEDIUM)

`Interc` is declared `real(8) :: Interc=0.0` (implicit SAVE) and is only
assigned inside an inner `if (LAI > eps)` guard within the interception
block (lines 443 and 448). When `LAI <= eps` (75 out of 1697 days in the
Qvidja dataset have `LAI = 0`), `Interc` retains the stale value from the
most recent call with `LAI > eps`. This stale `Interc` then:

1. Gets added to canopy storage: `W = W + Interc` (line 453)
2. Gets subtracted from throughfall: `Trfall = Prec + Unload - Interc` (line 462)

The net effect is phantom interception: water is trapped in the canopy
(which has zero LAI and should not intercept anything) and subtracted from
throughfall. Since W is typically 0 when LAI = 0, the stale Interc may also
drive W negative before being clamped by `CanopyEvap = min(erate, W + eps)`.

The JAX implementation correctly handles this with explicit `0.0` defaults:
```python
interc_snow = jnp.where(lai > eps, ..., 0.0)
interc_rain = jnp.where(lai > eps, ..., 0.0)
```

## Fix

Remove declaration initializers and add explicit initialization at the top of
the subroutine body:

```fortran
! Replace:
real(8)  :: Unload=0.0, Interc=0.0, Melt=0.0, Freeze=0.0, Trfall=0.0
real(8)  :: CanopyEvap=0.0, PotInfil=0.0

! With:
real(8)  :: Unload, Interc, Melt, Freeze, Trfall
real(8)  :: CanopyEvap, PotInfil

Unload = 0.0
Interc = 0.0
Melt = 0.0
Freeze = 0.0
Trfall = 0.0
CanopyEvap = 0.0
PotInfil = 0.0
```

This removes the implicit SAVE attribute from all local flux variables,
ensuring fresh initialization on every call.

## JAX implementation

The JAX port at `packages/svmc-jax/src/svmc_jax/water/canopy_soil.py` line 242
already handles this correctly:

```python
unload = jnp.where(tc >= tmin, jnp.maximum(w - wmax_tot, 0.0), 0.0)
```

Unload is explicitly set to 0.0 when `tc < tmin`. No SAVE attribute exists in
Python/JAX, so the JAX implementation represents the intended behavior.
