#!/usr/bin/env python3
"""
build_executable.py - Package Cortex as a single-file executable.

Produces:
    Windows  -> dist/Cortex.exe
    Linux    -> dist/Cortex
    macOS    -> dist/Cortex.app (or dist/Cortex)

Usage:
    pip install pyinstaller
    python build_executable.py

The resulting binary bundles Python and all dependencies. Users still
need to install Ollama separately (the LLMs are too large to ship).

Pass --debug to keep the console window visible (useful for diagnosing
startup errors). Pass --onedir for a faster-launching folder build
instead of single-file (single-file unpacks to temp on every launch
which adds ~2-3 seconds).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "cortex_launcher.py"
APP_NAME = "Cortex"


def check_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("ERROR: PyInstaller not installed.")
        print("  Run: pip install pyinstaller")
        sys.exit(1)


def check_entry() -> None:
    if not ENTRY.exists():
        print(f"ERROR: {ENTRY.name} not found in {ROOT}")
        print("  Make sure cortex.py and cortex_launcher.py are alongside this script.")
        sys.exit(1)
    if not (ROOT / "cortex.py").exists():
        print(f"ERROR: cortex.py not found in {ROOT}")
        sys.exit(1)


def kill_running_cortex() -> None:
    """Stop any running Cortex.exe so its dist/ files can be replaced.
    Quiet on Linux/macOS where it's rarely needed."""
    if platform.system() != "Windows":
        return
    try:
        # /F forces, /IM matches by image name; /T also kills child processes.
        # We don't care about output or exit code — if nothing is running, fine.
        subprocess.run(
            ["taskkill", "/F", "/IM", f"{APP_NAME}.exe", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        # Brief pause so Windows fully releases file handles
        import time
        time.sleep(0.5)
    except Exception:
        pass


def clean() -> None:
    """Remove previous build artifacts."""
    for path in (ROOT / "build", ROOT / "dist", ROOT / f"{APP_NAME}.spec"):
        if not path.exists():
            continue
        print(f"  removing {path.name}")
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except PermissionError as e:
            print(f"\nERROR: could not remove {path}: {e}")
            print(
                "  Something is still using files in that folder. "
                "Close any running Cortex window/process and try again.\n"
                "  PowerShell: Stop-Process -Name Cortex -Force"
            )
            sys.exit(1)


def build(onefile: bool, debug: bool) -> int:
    """Invoke PyInstaller with the right flags for the current platform."""
    system = platform.system()
    args: list[str] = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
    ]

    if onefile:
        args.append("--onefile")
    else:
        args.append("--onedir")

    # Hide the console window on Windows release builds. Keep it visible
    # in debug mode so users can copy error messages.
    if system == "Windows" and not debug:
        args.append("--noconsole")
    if debug:
        args.append("--console")

    # Hidden imports — packages PyInstaller's static analysis sometimes
    # misses because they're loaded lazily by FastAPI / uvicorn / ollama.
    hidden = [
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "ollama",
        "pypdf",
        "ebooklib",
        "ebooklib.epub",
        "bs4",
        "docx",
    ]
    for mod in hidden:
        args += ["--hidden-import", mod]

    # Optional icon (place icon.ico for Windows, icon.icns for macOS,
    # icon.png for Linux alongside this script).
    icon_map = {"Windows": "icon.ico", "Darwin": "icon.icns", "Linux": "icon.png"}
    icon_file = ROOT / icon_map.get(system, "")
    if icon_file.exists():
        args += ["--icon", str(icon_file)]

    args.append(str(ENTRY))

    print("\n" + "=" * 60)
    print(f"Building {APP_NAME} for {system}")
    print(f"  mode: {'onefile' if onefile else 'onedir'}")
    print(f"  debug: {debug}")
    print("=" * 60 + "\n")

    return subprocess.call(args, cwd=ROOT)


def report_output(onefile: bool) -> None:
    system = platform.system()
    dist = ROOT / "dist"
    if not dist.exists():
        print("\nBuild produced no dist/ directory — see errors above.")
        return

    if onefile:
        binary = dist / (f"{APP_NAME}.exe" if system == "Windows" else APP_NAME)
        if binary.exists():
            size_mb = binary.stat().st_size / (1024 * 1024)
            print(f"\nBuild succeeded: {binary}  ({size_mb:.1f} MB)")
        else:
            print(f"\nExpected binary not found at {binary}")
    else:
        folder = dist / APP_NAME
        if folder.exists():
            print(f"\nBuild succeeded: {folder}/")
            print(f"  Distribute the entire folder. Entry: {folder / APP_NAME}")
        else:
            print(f"\nExpected folder not found at {folder}")

    print("\nUsers still need to install Ollama separately:")
    print("  https://ollama.com")
    print("Then pull the models the first time:")
    print("  ollama pull qwen2.5:7b")
    print("  ollama pull nomic-embed-text")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Build {APP_NAME} executable")
    parser.add_argument("--onedir", action="store_true",
                        help="Build a folder distribution instead of single-file")
    parser.add_argument("--debug", action="store_true",
                        help="Keep console window visible for diagnostics")
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip cleaning previous build artifacts")
    args = parser.parse_args()

    check_pyinstaller()
    check_entry()

    if not args.no_clean:
        print("Cleaning previous builds...")
        kill_running_cortex()
        clean()

    rc = build(onefile=not args.onedir, debug=args.debug)
    if rc != 0:
        print(f"\nPyInstaller exited with code {rc}")
        return rc

    report_output(onefile=not args.onedir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
