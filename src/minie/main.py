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
    doc = sub.add_parser("doctor", help="Check mic, models, Anthropic key, and Cua Driver")
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
    from minie.audio.capture import MicStream

    store = StatusStore()
    engine = Engine(config, store.set)
    mic = MicStream(config)
    if no_menubar:
        try:
            engine.run(mic)
        except KeyboardInterrupt:
            _LOG.info("bye")
        finally:
            mic.stop()
        return

    boot = threading.Thread(target=engine.prepare, name="minie-prepare", daemon=True)
    boot.start()
    loop_started = threading.Event()

    def on_tick() -> None:
        if not engine.prepared.is_set():
            return
        if not mic.started:
            try:
                mic.start()
            except Exception:
                _LOG.exception("microphone failed to start")
                return
        if not loop_started.is_set():
            loop_started.set()
            threading.Thread(target=engine.loop, args=(mic,), name="minie-engine", daemon=True).start()

    try:
        start_menubar(store, engine.stop, on_tick)
    except RuntimeError:
        _LOG.warning("menu bar unavailable; running in the terminal")
        engine.run(mic)
    finally:
        engine.stop()
        mic.stop()


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
    check("ANTHROPIC_API_KEY", True, "set — Haiku plans CUA steps" if config.anthropic_api_key else "unset — GUI computer-use will refuse")
    check("TYPESAFE_API_KEY", True, "optional; Haiku picks AX targets until Jev is available")
    check("GEMINI_API_KEY", True, "optional; default CUA loop does not use it")
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
