"""Render the app icon to packaging/ as PNGs and a macOS .icns.

Run with: uv run python scripts/make_icon.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "packaging"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    # Qt offscreen so this runs without a display.
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from trafo.ui import theme  # noqa: E402

    app = QApplication([])  # noqa: F841
    base = theme.make_icon(1024)
    png_1024 = OUT / "icon_1024.png"
    base.pixmap(1024, 1024).save(str(png_1024))

    if sys.platform != "darwin":
        print(f"Saved {png_1024} (skipping .icns on non-macOS).")
        return 0

    iconset = OUT / "trafo.iconset"
    iconset.mkdir(exist_ok=True)
    for size in (16, 32, 64, 128, 256, 512, 1024):
        for scale, suffix in ((1, ""), (2, "@2x")):
            px = size * scale
            name = f"icon_{size}x{size}{suffix}.png"
            theme.make_icon(px).pixmap(px, px).save(str(iconset / name))
    icns = OUT / "trafo.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    print(f"Saved {icns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
