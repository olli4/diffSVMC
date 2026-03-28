# Third-Party Notices for svmc-ref

This package stages selected upstream Fortran files so the reference harness can be
built with `fpm`. The staged copies preserve upstream notices. License provenance is
recorded here so the staged tree is reviewable on its own. The complete staged file
set is defined in `packages/svmc-ref/staged-sources.json`.

## Upstream SVMC sources

The following staged files come from `vendor/SVMC/src/` and inherit the upstream
SVMC project license:

- `allocation.f90`
- `phydro_mod.f90`
- `readctrl_mod.f90`
- `readsoilpara_mod.f90`
- `readvegpara_mod.f90`
- `water_mod.f90`
- `yasso.f90`

Upstream project: `vendor/SVMC/`

Upstream license: MIT, see `vendor/SVMC/LICENSE`.

## L-BFGS-B support files

The following staged files are copied from `vendor/SVMC/src/` but originate from the
L-BFGS-B distribution referenced by the upstream file headers:

- `blas.f`
- `linpack.f`
- `lbfgsb.f`
- `timer.f`

Upstream project page: <https://users.iems.northwestern.edu/~nocedal/lbfgsb.html>

The upstream project page states that L-BFGS-B version 3.0 is released under the
"New BSD License" / BSD-3-Clause license. The local license text is recorded in
`packages/svmc-ref/LICENSES/BSD-3-Clause-L-BFGS-B.txt`.

The same upstream page also asks that publications or commercial products using the
software cite the L-BFGS-B references listed there.

## Yasso20 source

The staged file `yassofortran20.f90` is copied from `vendor/SVMC/src/yassofortran20.f90`.
Its source header states:

- Copyright (C) 2020 Finnish Meteorological Institute, Janne Pusa
- License: GNU General Public License, version 3 or later

The staged copy keeps that header unchanged. The canonical GPL text is available at
<https://www.gnu.org/licenses/gpl-3.0.txt>.