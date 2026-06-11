# PyInstaller spec for the Trafo macOS app bundle.
# Build:  uv run pyinstaller packaging/trafo.spec --noconfirm
# Output: dist/Trafo.app

import os

from PyInstaller.utils.hooks import collect_all

# A stable signing identity makes macOS permission grants survive rebuilds
# (ad-hoc signatures change every build, so TCC treats each build as a new
# app). Create a self-signed "Code Signing" certificate in Keychain Access,
# then build with TRAFO_CODESIGN_IDENTITY=<cert name>.
SIGN_ID = os.environ.get("TRAFO_CODESIGN_IDENTITY")

datas, binaries, hiddenimports = [], [], []

# mediapipe ships .tflite/.binarypb model assets and native libs that the
# default analysis misses; pull everything in.
for pkg in ("mediapipe", "cv2", "pynput"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

block_cipher = None

a = Analysis(
    ["launch.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["trafo"],
    hookspath=[],
    runtime_hooks=[],
    # matplotlib must NOT be excluded: mediapipe.tasks.python.vision imports
    # it at module level, so stripping it breaks the gaze pipeline in the
    # frozen app ("No module named 'matplotlib'").
    excludes=["tkinter"],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Trafo",
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    codesign_identity=SIGN_ID,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Trafo")

app = BUNDLE(
    coll,
    name="Trafo.app",
    icon="trafo.icns",
    bundle_identifier="com.trafo.app",
    info_plist={
        "CFBundleName": "Trafo",
        "CFBundleDisplayName": "Trafo",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        # Menu-bar (agent) app: lives in the tray, no permanent Dock icon.
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
        # Shown in the macOS camera permission prompt.
        "NSCameraUsageDescription":
            "Trafo uses the camera to track your eye gaze and focus the window you look at.",
    },
)
