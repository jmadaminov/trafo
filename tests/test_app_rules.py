import plistlib

from trafo.ui.app_rules import app_name_from_bundle


def _make_bundle(tmp_path, file_name, info: dict | None):
    bundle = tmp_path / file_name
    contents = bundle / "Contents"
    contents.mkdir(parents=True)
    if info is not None:
        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump(info, f)
    return bundle


def test_prefers_cfbundlename_over_file_name(tmp_path):
    # "Visual Studio Code.app" owns its windows as "Code".
    bundle = _make_bundle(
        tmp_path, "Visual Studio Code.app",
        {"CFBundleName": "Code", "CFBundleExecutable": "Electron"},
    )
    assert app_name_from_bundle(str(bundle)) == "Code"


def test_falls_back_to_executable_then_stem(tmp_path):
    bundle = _make_bundle(tmp_path, "Foo.app", {"CFBundleExecutable": "FooBin"})
    assert app_name_from_bundle(str(bundle)) == "FooBin"
    bare = _make_bundle(tmp_path, "Bare.app", None)  # no Info.plist at all
    assert app_name_from_bundle(str(bare)) == "Bare"
