"""Command-line entry point: `trafo <command>`."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trafo", description="Gaze-driven window focus")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="interactive demos for testing each component")
    demo.add_argument(
        "name", choices=["camera", "landmarks", "gaze", "dot"], help="which demo to run"
    )

    sub.add_parser("app", help="launch the Trafo app window")
    sub.add_parser("calibrate", help="launch the app and start calibration immediately")

    windows = sub.add_parser("windows", help="inspect and control desktop windows")
    windows_sub = windows.add_subparsers(dest="windows_command", required=True)
    windows_sub.add_parser("list", help="list visible windows, frontmost first")
    focus = windows_sub.add_parser("focus", help="focus the first window matching a substring")
    focus.add_argument("query", help="case-insensitive substring of window title or app name")

    args = parser.parse_args(argv)

    if args.command == "windows":
        return _windows_command(args)

    if args.command == "demo" and args.name in ("camera", "landmarks", "gaze"):
        from . import demos

        getattr(demos, f"{args.name}_demo")(args.camera)
        return 0

    from .app import run_app

    if args.command == "app":
        return run_app(args.camera)
    if args.command == "calibrate":
        return run_app(args.camera, auto_calibrate=True)
    if args.command == "demo":  # dot
        return run_app(args.camera, overlay_on=True)
    return 2


def _windows_command(args) -> int:
    from .winmgr import get_window_manager

    wm = get_window_manager()
    for problem in wm.permissions_missing():
        print(f"⚠ missing permission: {problem}")
        if hasattr(wm, "request_permissions"):
            wm.request_permissions()

    if args.windows_command == "list":
        windows = wm.list_windows()
        if not windows:
            print("No windows found.")
            return 1
        print(f"{'#':>3}  {'app':<24} {'rect (x, y, w, h)':<30} title")
        for i, w in enumerate(windows):
            rect = ", ".join(f"{v:.0f}" for v in w.rect)
            print(f"{i:>3}  {w.app[:24]:<24} {rect:<30} {w.title[:50]}")
        return 0

    # focus
    query = args.query.lower()
    for w in wm.list_windows():
        if query in w.title.lower() or query in w.app.lower():
            ok = wm.focus(w)
            print(f"{'Focused' if ok else 'FAILED to focus'}: {w.app} — {w.title!r}")
            return 0 if ok else 1
    print(f"No window matching {args.query!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
