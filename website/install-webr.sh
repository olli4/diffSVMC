#!/usr/bin/env bash
# One-time fix: reclaim ownership of website/public/{bin,src} so that
# future builds can write there without sudo.
#
# Run once:  sudo website/install-webr.sh
# Then:      ./website/build-webr.sh   (no sudo needed)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/website/public"

CALLER_UID="${SUDO_UID:-$(id -u)}"
CALLER_GID="${SUDO_GID:-$(id -g)}"

changed=0
for d in "$OUT/bin" "$OUT/src"; do
  if [[ -d "$d" ]] && [[ "$(stat -c %u "$d")" != "$CALLER_UID" ]]; then
    chown -R "$CALLER_UID:$CALLER_GID" "$d"
    echo "Fixed ownership: $d"
    changed=1
  fi
done

if [[ "$changed" -eq 0 ]]; then
  echo "Nothing to fix — website/public/{bin,src} already owned by uid $CALLER_UID."
fi
