from minie.agent.adapters.base import Adapter
from minie.agent.adapters.facetime import FaceTimeAdapter
from minie.agent.adapters.open_app import OpenAppAdapter

__all__ = [
    "Adapter",
    "FaceTimeAdapter",
    "OpenAppAdapter",
    "native_adapters",
]


def native_adapters() -> list[Adapter]:
    """OS-level shortcuts only. App workflows go through the AX structure loop."""
    return [FaceTimeAdapter(), OpenAppAdapter()]
