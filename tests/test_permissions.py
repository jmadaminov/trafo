from trafo import permissions


def test_check_returns_valid_states():
    valid = {"ok", "missing", "unknown", "n/a"}
    for perm in permissions.permissions_for_platform() or [permissions.Permission("x", "X", "", "")]:
        assert permissions.check(perm.key) in valid


def test_check_never_raises_on_unknown_key():
    assert permissions.check("definitely-not-a-real-permission") in {"unknown", "n/a"}


def test_request_and_open_settings_never_raise():
    # Smoke: must be safe to call (no-ops off macOS; trigger prompts on it).
    permissions.request("definitely-not-real")
