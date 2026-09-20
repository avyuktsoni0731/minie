from minie.agent.adapters.base import Adapter
from minie.agent.adapters.calculator import CalculatorAdapter
from minie.agent.adapters.facetime import FaceTimeAdapter
from minie.agent.adapters.open_app import OpenAppAdapter

__all__ = [
    "Adapter",
    "CalculatorAdapter",
    "FaceTimeAdapter",
    "OpenAppAdapter",
    "native_adapters",
]


def native_adapters() -> list[Adapter]:
    """Order matters: specific outcomes before generic open-app."""
    return [FaceTimeAdapter(), CalculatorAdapter(), OpenAppAdapter()]
