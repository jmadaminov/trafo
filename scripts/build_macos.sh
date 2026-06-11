#!/usr/bin/env bash
# Build the Trafo macOS .app bundle.
#
# Usage: ./scripts/build_macos.sh
# Output: dist/Trafo.app
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Generating icon"
uv run python scripts/make_icon.py

echo "==> Building app bundle with PyInstaller"
uv run pyinstaller packaging/trafo.spec --noconfirm --clean

echo
echo "==> Done: dist/Trafo.app"
echo "    First launch: right-click ▸ Open (unsigned beta build)."
echo "    Grant Camera, Screen Recording, Accessibility and Input Monitoring"
echo "    when prompted, then quit and reopen if a permission doesn't take effect."
