#!/usr/bin/env python3
"""
cortex_launcher.py - Entry point for the packaged Cortex executable.

Wraps cortex.py with desktop-app niceties:
  - Pings Ollama before starting; shows a clear message if it's missing
  - Opens the user's default browser to the chat UI automatically
  - Logs to a file alongside the executable so users can share errors
  - Exits cleanly on Ctrl+C / window close

Run directly during development:
    python cortex_launcher.py

For the bundled build, point PyInstaller at this file (build_executable.py
already does that).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback


# === Windowless build safety ==============================================
# When PyInstaller builds with --noconsole on Windows, sys.stdout and
# sys.stderr are set to None. Anything that calls .isatty(), .write(),
# or .flush() on them crashes — most notably uvicorn's logging formatter,
# which calls sys.stderr.isatty() during config to decide on ANSI colors.
# Replace missing streams with a minimal file-like object BEFORE any
# import that might touch them.

class _NullStream:
    """File-like sink for environments where stdout/stderr are unavailable."""
    def write(self, *_args, **_kwargs):
        return 0
    def flush(self):
        pass
    def isatty(self):
        return False
    def fileno(self):
        raise OSError("no fileno")
    def close(self):
        pass


if sys.stdout is None:
    sys.stdout = _NullStream()
if sys.stderr is None:
    sys.stderr = _NullStream()


class _StreamToLogger:
    """File-like proxy that forwards writes to a logger. Used to capture
    print() output from cortex.py into the log file once logging is up."""
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        if not message:
            return 0
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self._logger.log(self._level, line)
        return len(message)

    def flush(self):
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.rstrip())
            self._buffer = ""

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("no fileno")

    def close(self):
        self.flush()


import webbrowser   # noqa: E402  -- imported after stream guards
from pathlib import Path



def _log_path() -> Path:
    """Where to write the launcher log. Next to the executable when frozen,
    or alongside this file in dev. Fall back to home dir if neither is writable."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent
    candidate = base / "cortex.log"
    try:
        candidate.touch(exist_ok=True)
        return candidate
    except (PermissionError, OSError):
        return Path.home() / "cortex.log"


def _setup_logging() -> Path:
    log_file = _log_path()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    # Once logging is up, redirect any null/no-op stdout/stderr to the
    # log file. This catches print() calls from cortex.py and from any
    # third-party code that writes diagnostics to stderr.
    if isinstance(sys.stdout, _NullStream):
        sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    if isinstance(sys.stderr, _NullStream):
        sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.WARNING)
    return log_file


def _check_ollama(host: str) -> tuple[bool, str]:
    """Return (ok, message). Tries the /api/tags endpoint with a short timeout."""
    try:
        import urllib.request
        url = f"{host.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return True, "Ollama is running."
            return False, f"Ollama responded with HTTP {resp.status}."
    except Exception as e:
        return False, str(e)


def _show_error_and_exit(title: str, message: str, log_file: Path) -> "None":
    """Best-effort: show a GUI dialog if tkinter is available, otherwise print."""
    full_message = f"{message}\n\nLog file: {log_file}"
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, full_message)
        root.destroy()
    except Exception:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        print(full_message)
        print("=" * 60)
        # On Windows double-click without console, this print is invisible —
        # the dialog above is the real fallback. The pause helps if a
        # console *is* attached.
        if sys.platform == "win32":
            try:
                input("\nPress Enter to exit...")
            except EOFError:
                pass
    sys.exit(1)


def _open_browser_when_ready(url: str, max_wait: float = 10.0) -> None:
    """Wait until the server accepts connections, then open the browser."""
    import urllib.request
    import urllib.error
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5)
            break
        except urllib.error.URLError:
            time.sleep(0.2)
        except Exception:
            time.sleep(0.2)
    try:
        webbrowser.open(url)
        logging.info("Opened browser at %s", url)
    except Exception as e:
        logging.warning("Could not open browser automatically: %s", e)


def main() -> int:
    log_file = _setup_logging()
    logging.info("Cortex launcher starting (frozen=%s)", getattr(sys, "frozen", False))

    # Allow the bundled binary to find its sibling cortex.py module when
    # PyInstaller unpacks it. PyInstaller adds _MEIPASS to sys.path
    # automatically; this is a safety net for onedir builds.
    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        if str(base_dir) not in sys.path:
            sys.path.insert(0, str(base_dir))

    # Import cortex *after* logging is configured so its prints land in the log.
    try:
        import cortex
    except Exception as e:
        logging.exception("Failed to import cortex module")
        _show_error_and_exit(
            "Cortex failed to start",
            f"Could not load the application module.\n\n{e}\n\n"
            "This usually means a dependency is missing in the build.",
            log_file,
        )
        return 1

    host = os.environ.get("CORTEX_HOST", "127.0.0.1")
    port = int(os.environ.get("CORTEX_PORT", "8000"))
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = f"http://{host}:{port}"

    # Ollama check — clear actionable message if not present.
    ok, msg = _check_ollama(ollama_host)
    if not ok:
        logging.error("Ollama not reachable at %s: %s", ollama_host, msg)
        _show_error_and_exit(
            "Ollama is required",
            "Cortex needs Ollama to run, but it isn't reachable.\n\n"
            f"Tried: {ollama_host}\n"
            f"Reason: {msg}\n\n"
            "1. Install Ollama from https://ollama.com\n"
            "2. Make sure it's running (it usually starts automatically).\n"
            "3. Pull the default models on first use:\n"
            "      ollama pull qwen2.5:7b\n"
            "      ollama pull nomic-embed-text\n\n"
            "Then launch Cortex again.",
            log_file,
        )
        return 1

    logging.info("Ollama OK at %s. Launching Cortex on %s", ollama_host, url)

    # Open the browser shortly after the server is ready.
    threading.Thread(
        target=_open_browser_when_ready, args=(url,), daemon=True
    ).start()

    # Hand off to uvicorn. Use an explicit log_config (no ColourizedFormatter)
    # to avoid any internal isatty checks even when streams are wrapped.
    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "uvicorn.error":  {"level": "WARNING"},
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    try:
        import uvicorn
        uvicorn.run(
            cortex.app,
            host=host,
            port=port,
            log_level="warning",
            log_config=uvicorn_log_config,
        )
    except KeyboardInterrupt:
        logging.info("Shutting down (Ctrl+C)")
        return 0
    except Exception:
        logging.exception("Uvicorn crashed")
        _show_error_and_exit(
            "Cortex crashed",
            "The server stopped unexpectedly. See the log file for details.",
            log_file,
        )
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Last-ditch handler so a crash before logging is set up still
        # produces *something* on screen.
        traceback.print_exc()
        if sys.platform == "win32":
            try:
                input("\nPress Enter to exit...")
            except EOFError:
                pass
        sys.exit(1)
