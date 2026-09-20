from __future__ import annotations

from abc import ABC, abstractmethod

from minie.agent.task import AdapterContext, AdapterResult, Task


class Adapter(ABC):
    name = "adapter"

    @abstractmethod
    def claim(self, instruction: str, ctx: AdapterContext | None = None) -> Task | None:
        """Return a Task if this native adapter should handle the instruction."""

    def speculate(self, instruction: str, ctx: AdapterContext) -> None:
        """Safe work from a still-growing transcript. Default: do nothing."""

    @abstractmethod
    def execute(self, task: Task, ctx: AdapterContext) -> AdapterResult:
        """Run the task. Must check success_criteria before reporting ok."""
