#!/usr/bin/env bash
# Build the SVMCwebr WASM R package locally using Docker,
# producing a CRAN-like repo in website/public/.
#
# Docker outputs to tmp/webr-staging/ first while running as the
# current host uid:gid, then copies across to website/public/.
#
# If website/public/{bin,src} are root-owned from a previous build,
# run the one-time fix first:  sudo website/install-webr.sh
#
# Usage:
#   ./website/build-webr.sh               # build via Docker
#   SKIP_WEBR=1 ./website/build-webr.sh   # skip (CI handles it)
set -euo pipefail

if [[ "${SKIP_WEBR:-}" == "1" ]]; then
  echo "SKIP_WEBR=1 — skipping WebR build"
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$REPO_ROOT/tmp/webr-staging"
OUT="$REPO_ROOT/website/public"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# If the CRAN repo already exists in public/, skip rebuild.
if [[ -f "$OUT/bin/emscripten/contrib/4.5/PACKAGES" ]]; then
  echo "WebR package already in website/public/ — skipping rebuild."
  echo "To force: rm -rf website/public/bin website/public/src"
  exit 0
fi

# ---- Check for root-owned leftovers before we start the slow Docker build ----
for d in "$STAGING" "$OUT/bin" "$OUT/src"; do
  if [[ -d "$d" ]] && [[ "$(stat -c %u "$d")" != "$(id -u)" ]]; then
    echo "ERROR: $d is not owned by you (owned by uid $(stat -c %u "$d"))."
    if [[ "$d" == "$STAGING" ]]; then
      echo "One-time fix:  sudo chown -R $HOST_UID:$HOST_GID tmp/webr-staging"
    else
      echo "One-time fix:  sudo website/install-webr.sh"
    fi
    exit 1
  fi
done

rm -rf "$STAGING"
mkdir -p "$STAGING" "$OUT"

# Pick a working docker command (rootless or sudo).
DOCKER=
if docker info &>/dev/null; then
  DOCKER=docker
elif sudo docker info &>/dev/null; then
  DOCKER="sudo docker"
else
  echo "Docker not available. To populate website/public/ manually:"
  echo "  tar xf tmp/artifact.tar -C website/public/"
  exit 1
fi

# Clean stale native build artifacts so the Docker cross-compile starts fresh.
rm -f "$REPO_ROOT"/packages/svmc-webr/src/*.o \
     "$REPO_ROOT"/packages/svmc-webr/src/*.mod \
     "$REPO_ROOT"/packages/svmc-webr/src/*.so

echo "Building SVMCwebr WASM package via Docker…"
$DOCKER run --rm \
  --user "$HOST_UID:$HOST_GID" \
  -e HOME=/tmp \
  -v "$REPO_ROOT":/work \
  -w /work \
  ghcr.io/r-wasm/webr:main \
  bash -c '
    mkdir -p /tmp/repo
    Rscript -e "rwasm::add_pkg(\"local::packages/svmc-webr\", repo_dir = \"/tmp/repo\")"
    cp -r /tmp/repo/* /work/tmp/webr-staging/
  '

# Copy from staging to website/public/ (user-owned after one-time fix).
cp -r "$STAGING"/* "$OUT"/

echo "WebR package built → website/public/"
ls -R "$OUT"
