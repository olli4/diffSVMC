#!/usr/bin/env bash
# Build the SVMCwebr WASM R package locally using Docker,
# producing a CRAN-like repo in website/public/.
#
# Usage:
#   ./website/build-webr.sh          # build via Docker
#   SKIP_WEBR=1 ./website/build-webr.sh  # skip (CI handles it)
set -euo pipefail

if [[ "${SKIP_WEBR:-}" == "1" ]]; then
  echo "SKIP_WEBR=1 — skipping WebR build"
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/website/public"

# If the CRAN repo already exists in public/, skip rebuild.
if [[ -f "$OUT/bin/emscripten/contrib/4.5/PACKAGES" ]]; then
  echo "WebR package already in website/public/ — skipping rebuild."
  echo "To force: rm -rf website/public/bin website/public/src"
  exit 0
fi

mkdir -p "$OUT"

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

echo "Building SVMCwebr WASM package via Docker…"
$DOCKER run --rm \
  -v "$REPO_ROOT":/work \
  -w /work \
  ghcr.io/r-wasm/webr:main \
  bash -c '
    mkdir -p /tmp/repo
    Rscript -e "rwasm::add_pkg(\"local::packages/svmc-webr\", repo_dir = \"/tmp/repo\")"
    cp -r /tmp/repo/* /work/website/public/
  '

echo "WebR package built → website/public/"
ls -R "$OUT"
