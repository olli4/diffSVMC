#!/usr/bin/env bash
# Re-sync allocation Fortran source from vendor/SVMC into the R package.
# Applies the real -> real(8) promotion for the three mixed-precision
# arguments in alloc_hypothesis_2 and removes readalloc_namelist.
#
# Usage: packages/svmc-webr/sync-sources.sh   (from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENDOR_SRC="$(cd "$SCRIPT_DIR/../../vendor/SVMC/src" && pwd)"
PKG_SRC="$SCRIPT_DIR/src"

echo "Syncing allocation.f90 from $VENDOR_SRC ..."

# Copy and patch: promote real -> real(8) for the three arguments
sed \
  -e '/^      real, intent(inout) :: litter_cleaf/s/^      real,/      real(8),/' \
  -e '/^      real, intent(inout) :: litter_croot/s/^      real,/      real(8),/' \
  -e '/^      real, intent(inout) :: compost/s/^      real,/      real(8),/' \
  "$VENDOR_SRC/allocation.f90" \
  > "$PKG_SRC/allocation.f90.tmp"

# Remove the readalloc_namelist subroutine (file I/O, incompatible with WebR)
# Keep the header comment about provenance
cat > "$PKG_SRC/allocation.f90" << 'HEADER'
! SVMC allocation module — R-package copy
!
! Provenance: vendor/SVMC/src/allocation.f90
! Modifications for R package compatibility:
!   1. litter_cleaf, litter_croot, compost in alloc_hypothesis_2 changed from
!      real to real(8). The vendor build uses -freal-4-real-8 so these are
!      effectively double precision in the reference model.
!   2. readalloc_namelist subroutine removed (does file I/O incompatible with
!      WebR; parameters are passed via R arguments instead).
!
! No numerical changes relative to the vendor build with -freal-4-real-8.
HEADER

# Extract the file but strip readalloc_namelist
python3 -c "
import re, sys
src = open('$PKG_SRC/allocation.f90.tmp').read()
# Remove readalloc_namelist subroutine
src = re.sub(
    r'   subroutine readalloc_namelist\b.*?end subroutine readalloc_namelist\n',
    '', src, flags=re.DOTALL)
sys.stdout.write(src)
" >> "$PKG_SRC/allocation.f90"

rm "$PKG_SRC/allocation.f90.tmp"
echo "Done. Review $PKG_SRC/allocation.f90 for correctness."
