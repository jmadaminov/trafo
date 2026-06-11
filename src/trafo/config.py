"""Config and data paths (~/.config/trafo on all platforms)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "trafo"
CALIBRATION_PATH = CONFIG_DIR / "calibration.json"
SETTINGS_PATH = CONFIG_DIR / "settings.json"


def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


@dataclass
class Settings:
    dwell_ms: int = 500
    learn_from_clicks: bool = True
    onboarded: bool = False

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_PATH.exists():
            try:
                known = {k: v for k, v in json.loads(SETTINGS_PATH.read_text()).items()
                         if k in cls.__dataclass_fields__}
                return cls(**known)
            except (json.JSONDecodeError, TypeError):
                pass
        return cls()

    def save(self) -> None:
        ensure_config_dir()
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2))
