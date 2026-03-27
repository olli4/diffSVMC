# aerodynamics zn_cap branch is unreachable dead code

**File:** `src/water_mod.f90`, subroutine `aerodynamics`  
**Severity:** Cosmetic (no functional impact)

## Summary

The normalized ground-height cap `min(zg1 / hc, 1.0)` on line 556 can never
take the capping branch because `zg1` is already clamped to `0.1 * hc` two
lines earlier.

## Code

```fortran
! line 550 — ground height capped at 10% of canopy height
zg1 = min(spafhy_para%zground, 0.1 * spafhy_para%hc)

! ...

! line 556 — normalize and cap at 1.0 (intended to prevent zg > hc)
zn = min(zg1 / spafhy_para%hc, 1.0)
```

After line 550, `zg1 ≤ 0.1 * hc`, so `zg1 / hc ≤ 0.1`.  The subsequent
`min(..., 1.0)` always selects the first argument.

## Suggestion

Either:

1. Remove the redundant cap:
   ```fortran
   zn = zg1 / spafhy_para%hc
   ```

2. Or, if the intent was to allow `zground > hc` in some configurations and
   only cap at the canopy top, remove the `0.1 * hc` clamp on line 550 and
   keep the `min(..., 1.0)` on line 556.

## Impact

None — the dead branch has no effect on computed values.

## Provenance

Found during a systematic differentiable port of SVMC to JAX and TypeScript.
Verified present in upstream `huitang-earth/SVMC` main branch.
