from __future__ import annotations

import argparse
import shutil
import sys
import threading

from minie.config import Config
from minie.engine import Engine
from minie.log import get_logger, setup_logging
from minie.ui.menubar import StatusStore, start_menubar

_LOG = get_logger()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="minie", description="Wake, speak, act on your Mac.")
    sub = parser.add_subparsers(dest="cmd")
    run_p = sub.add_parser("run", help="Start the Minie daemon (default)")
    run_p.add_argument("--no-menubar", action="store_true", help="Run in the terminal only")
    run_p.add_argument("--debug", action="store_true")
    doc = sub.add_parser("doctor", help="Check mic, models, Gemini key, and Cua Driver")
    doc.add_argument("--debug", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-menubar", action="store_true")
    args = parser.parse_args(argv)
    cmd = args.cmd or "run"
    config = Config()
    debug = bool(getattr(args, "debug", False) or config.debug)
    setup_logging(debug)
    if cmd == "doctor":
        raise SystemExit(_doctor(config))
    _run(config, no_menubar=bool(args.no_menubar))


def _run(config: Config, *, no_menubar: bool) -> None:
    store = StatusStore()
    engine = Engine(config, store.set)
    if no_menubar:
        try:
            engine.run()
        except KeyboardInterrupt:
            _LOG.info("bye")
        return
    thread = threading.Thread(target=engine.run, name="minie-engine", daemon=True)
    thread.start()
    try:
        start_menubar(store, engine.stop)
    except RuntimeError:
        _LOG.warning("menu bar unavailable; running in the terminal")
        engine.run()


def _doctor(config: Config) -> int:
    from minie.computer.cua import CuaDriver

    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        mark = "ok" if passed else "MISSING"
        if not passed:
            ok = False
        extra = f" — {detail}" if detail else ""
        print(f"[{mark}] {label}{extra}")

    check("Python 3.12", sys.version_info[:2] == (3, 12), sys.version.split()[0])
    try:
        import mlx  # noqa: F401

        check("mlx", True)
    except Exception as exc:
        check("mlx", False, str(exc))
    try:
        import mlx_whisper  # noqa: F401

        check("mlx-whisper", True)
    except Exception as exc:
        check("mlx-whisper", False, str(exc))
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        check("microphone devices", len(devices) > 0, f"{len(devices)} devices")
    except Exception as exc:
        check("sounddevice / PortAudio", False, str(exc))
    check("GEMINI_API_KEY", bool(config.gemini_api_key), "set in .env" if config.gemini_api_key else "copy .env.example to .env")
    cua_path = shutil.which(config.cua_bin)
    check("cua-driver on PATH", cua_path is not None, cua_path or "install from https://cua.ai/cua-driver")
    if cua_path:
        try:
            CuaDriver(config).list_apps()
            check("Cua Driver daemon + permissions", True)
        except Exception as exc:
            check(
                "Cua Driver daemon + permissions",
                False,
                "run: open -n -g -a CuaDriver --args serve && cua-driver permissions grant "
                f"({exc})",
            )
    print()
    print('Wake word: "Hey Minie"')
    print("Grant Microphone to Terminal/Minie. Grant Accessibility + Screen Recording to CuaDriver.app.")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
