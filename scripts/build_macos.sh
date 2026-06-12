#!/usr/bin/env bash
# Build the Trafo macOS .app bundle and a drag-to-install DMG.
#
# Usage:
#   ./scripts/build_macos.sh             build dist/Trafo.app + dist/Trafo-<ver>.dmg
#   ./scripts/build_macos.sh --install   also copy the app into /Applications
#
# Signing (recommended so permission grants survive rebuilds):
#   TRAFO_CODESIGN_IDENTITY=trafo-dev ./scripts/build_macos.sh
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION="$(grep -m1 'CFBundleShortVersionString' packaging/trafo.spec | sed 's/[^0-9.]*//g')"
DMG="dist/Trafo-${VERSION}.dmg"

echo "==> Generating icon"
uv run python scripts/make_icon.py

echo "==> Building app bundle with PyInstaller"
if [[ -n "${TRAFO_CODESIGN_IDENTITY:-}" ]]; then
  echo "    signing with identity: ${TRAFO_CODESIGN_IDENTITY}"
else
  echo "    NOTE: ad-hoc signature — macOS forgets permission grants on every"
  echo "    rebuild. See README 'Beta signing caveat' for the one-time fix."
fi
uv run pyinstaller packaging/trafo.spec --noconfirm --clean

echo "==> Creating ${DMG}"
# A mounted copy of a previous DMG makes hdiutil fail — detach it first.
if [[ -d "/Volumes/Trafo" ]]; then
  hdiutil detach "/Volumes/Trafo" || true
fi
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
ditto dist/Trafo.app "$STAGING/Trafo.app"
ln -s /Applications "$STAGING/Applications"
rm -f "$DMG"
if ! hdiutil create -volname "Trafo" -srcfolder "$STAGING" -ov -format UDZO "$DMG"; then
  echo "    WARNING: DMG creation failed — dist/Trafo.app itself is fine."
fi

if [[ "${1:-}" == "--install" ]]; then
  echo "==> Installing to /Applications/Trafo.app"
  rm -rf /Applications/Trafo.app
  ditto dist/Trafo.app /Applications/Trafo.app
fi

echo
echo "==> Done"
echo "    App:  dist/Trafo.app"
echo "    DMG:  ${DMG}  (share this — open it and drag Trafo into Applications)"
echo "    First launch: right-click Trafo.app ▸ Open (unsigned beta build)."
