# SVMCwebr — SVMC model for WebR

R package wrapping the full SVMC (Simple Vegetation Model of Carbon) Fortran
model for use in [WebR](https://docs.r-wasm.org/webr/latest/).
This enables running the original Fortran code in the browser for
side-by-side comparison with the `jax-js-nonconsuming` TypeScript port.

The package includes the complete daily integration loop: P-Hydro
photosynthesis, SpaFHy canopy and soil water balance, carbon allocation,
and YASSO20 soil carbon decomposition.

## Provenance

The Fortran source in `src/` derives from `vendor/SVMC/src/`
(which itself derives from [huitang-earth/SVMC](https://github.com/huitang-earth/SVMC)).

Modifications from the vendor source:

1. **`real` → `real(8)` promotion** for three arguments (`litter_cleaf`,
   `litter_croot`, `compost`) in `alloc_hypothesis_2`.  The vendor Makefile
   uses `-freal-4-real-8` so these are effectively `double precision` in
   the reference build.  Explicit `real(8)` avoids compiler-specific flags
   and matches the reference numerical behavior.

2. **Namelist I/O removed** — `readalloc_namelist`, `readsoilpara_namelist`,
   `readvegpara_namelist`, and `readctrl_namelist` are replaced by no-op stubs.
   Parameters are instead passed as R function arguments.

3. **YASSO initialization climate** — YASSO20 soil pool initialization
   uses forcing-derived temperature and precipitation defaults instead of
   fixed stub values, improving spin-up accuracy for site-specific runs.

4. **WASM compatibility** — `print`, `write`, `stop`, and `error stop`
   statements are removed or replaced with silent returns. Key input
   validation (YASSO fraction bounds, array lengths) is performed on the
   R side before calling Fortran.

5. **`allocation_wrappers.f90`** provides standalone (non-module) subroutines
   with flat scalar argument lists callable via R's `.Fortran()` interface.

## Exported Functions

### `svmc_run(...)`

Runs the full SVMC integration loop for a given period: hourly
photosynthesis and water balance, daily carbon allocation and YASSO20
soil decomposition.  Returns a list with daily output time series
(GPP, NEE, soil carbon pools, water balance, etc.).
See `?svmc_run` for argument details.

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

## Building

### WASM (WebR) via Docker

The `website/build-webr.sh` script builds the package to WebAssembly
using the `ghcr.io/r-wasm/webr:main` Docker image. Docker output is
staged through `tmp/webr-staging/` and copied to `website/public/`:

```bash
# From the repo root:
pnpm build              # builds WASM package + Vite site
pnpm -C website dev     # dev server at http://localhost:5173
```

If `website/public/{bin,src}` are root-owned from a previous Docker build,
run the one-time ownership fix first:

```bash
sudo website/install-webr.sh
```

The script auto-detects whether Docker needs `sudo`. If Docker is not
available, you can seed `website/public/` from a CI artifact instead:

```bash
tar xf tmp/artifact.tar -C website/public/
```

### Native R (for local testing)

Requires R ≥ 4.3 and gfortran. On Ubuntu/Debian:

```bash
sudo apt-get install r-base r-base-dev
```

Build and install:

```bash
R CMD build packages/svmc-webr
mkdir -p tmp/R-lib
R CMD INSTALL --library=tmp/R-lib SVMCwebr_0.1.0.tar.gz
```

Test:

```bash
R_LIBS=tmp/R-lib R -e '
library(SVMCwebr)
result <- alloc_hypothesis_2(
  temp_day = 15, gpp_day = 3e-7, leaf_rdark_day = 3e-8,
  cleaf = 0.1, cstem = 0.02, croot = 0.05,
  pft_type_code = 1L  # grass
)
str(result)
'
```

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

## Adding Fortran Entry Points

To expose additional Fortran subroutines to R:

1. Add wrapper subroutines in `src/` with flat argument lists.
2. Register the new Fortran entry points in `src/init.c`.
3. Add R wrapper functions in `R/`.
4. Export in `NAMESPACE`.
5. Update Makevars module dependency rules.
