"""Frozen-app entry point (run as a top-level script by PyInstaller).

Uses an absolute import because, unlike `python -m trafo`, this file is
executed without package context.
"""

from trafo.app import run_app

if __name__ == "__main__":
    raise SystemExit(run_app())
