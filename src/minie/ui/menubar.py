from __future__ import annotations

from collections.abc import Callable

from minie.config import STATUS_IDLE

try:
    import rumps
except Exception:  # pragma: no cover - optional GUI
    rumps = None  # type: ignore[assignment]


class StatusStore:
    def __init__(self) -> None:
        self.value = STATUS_IDLE

    def set(self, status: str) -> None:
        self.value = status


def start_menubar(
    store: StatusStore,
    on_quit: Callable[[], None],
    on_tick: Callable[[], None] | None = None,
) -> None:
    if rumps is None:
        raise RuntimeError("rumps is not available")

    class MinieBar(rumps.App):
        def __init__(self) -> None:
            super().__init__("Minie", title=_title(store.value), quit_button="Quit Minie")
            self.menu = ["Minie v1 — Hey Minie"]
            self._last = store.value
            self.timer = rumps.Timer(self._tick, 0.25)
            self.timer.start()

        def _tick(self, _timer: object) -> None:
            if on_tick is not None:
                on_tick()
            if store.value != self._last:
                self._last = store.value
                self.title = _title(store.value)

        def quit_application(self, sender: object = None) -> None:  # noqa: ANN001
            on_quit()
            super().quit_application(sender)

    MinieBar().run()


def _title(status: str) -> str:
    return f"M · {status}"
