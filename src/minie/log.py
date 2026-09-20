from __future__ import annotations

import logging
import sys

_LOG = logging.getLogger("minie")


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    if not _LOG.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        _LOG.addHandler(handler)
    _LOG.setLevel(level)


def get_logger() -> logging.Logger:
    return _LOG
