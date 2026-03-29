# SVMCwebr — SVMC allocation model for WebR

R package wrapping the SVMC (Simple Vegetation Model of Carbon) Fortran
allocation submodel for use in [WebR](https://docs.r-wasm.org/webr/latest/).
This enables running the original Fortran allocation code in the browser
for side-by-side comparison with the `jax-js-nonconsuming` TypeScript port.

## Provenance

The Fortran source in `src/` derives from `vendor/SVMC/src/allocation.f90`
(which itself derives from [huitang-earth/SVMC](https://github.com/huitang-earth/SVMC)).

Modifications from the vendor source:

1. **`real` → `real(8)` promotion** for three arguments (`litter_cleaf`,
   `litter_croot`, `compost`) in `alloc_hypothesis_2`.  The vendor Makefile
   uses `-freal-4-real-8` so these are effectively `double precision` in
   the reference build.  Explicit `real(8)` avoids compiler-specific flags
   and matches the reference numerical behavior.

2. **`readalloc_namelist` removed** — it reads parameters from a file,
   which is incompatible with WebR's virtual filesystem.  Parameters are
   instead passed as R function arguments.

3. **`readvegpara_stub.f90`** replaces the full `readvegpara_mod.f90` —
   provides only the `pft_type` character variable needed by the allocation
   module, without any file I/O subroutines.

4. **`allocation_wrappers.f90`** provides standalone (non-module) subroutines
   `r_alloc_h2` and `r_invert_alloc` with flat scalar argument lists
   callable via R's `.Fortran()` interface.

No numerical changes relative to the vendor build with `-freal-4-real-8`.

## Exported Functions

### `alloc_hypothesis_2(...)`

Runs carbon allocation (growth, respiration, turnover, management events)
for one daily time step.  See `?alloc_hypothesis_2` for argument details.

### `invert_alloc(...)`

Derives allometric parameters (`cratio_leaf` or `turnover_cleaf`) from
observed LAI changes.  See `?invert_alloc` for argument details.

### PFT type codes

| Code | Plant Functional Type |
|------|-----------------------|
| `0`  | Other                 |
| `1`  | Grass                 |
| `2`  | Oat                   |

## WebR Usage

Once the CI workflow builds and deploys the WASM binary to GitHub Pages,
install from the [WebR REPL](https://webr.r-wasm.org/latest/) or from
JavaScript:

```r
install.packages("SVMCwebr",
  repos = c(
    "https://<owner>.github.io/diffSVMC/",
    "https://repo.r-wasm.org/"
  ))
library(SVMCwebr)

result <- alloc_hypothesis_2(
  temp_day = 15, gpp_day = 5e-6, leaf_rdark_day = 1e-7,
  cleaf = 0.05, cstem = 0.1, croot = 0.03,
  pft_type_code = 1L  # grass
)
str(result)
```

## Re-syncing from Vendor

If the vendor Fortran source is updated, re-sync the R package copy:

```bash
# From repo root:
packages/svmc-webr/sync-sources.sh
```

## Extending to Other Submodels

To add more SVMC submodels (e.g., phydro, spafhy, yasso):

1. Copy the needed `.f90` sources into `src/`, applying any `real→real(8)`
   or I/O-removal fixes.
2. Add corresponding wrapper subroutines in `src/` with flat argument lists.
3. Register the new Fortran entry points in `src/init.c`.
4. Add R wrapper functions in `R/`.
5. Export in `NAMESPACE`.
6. Update Makevars module dependency rules.
