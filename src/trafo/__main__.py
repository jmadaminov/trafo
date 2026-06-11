"""Entry point for the packaged app (`python -m trafo` / the .app bundle)."""

from .app import run_app

if __name__ == "__main__":
    raise SystemExit(run_app())
