"""Gerenciador de estado de saúde por componente da aplicação.

Cada subsistema (RAG, Agent, DriftDetector, Guardrails) tem seu próprio
ComponentHealth. O lifespan do FastAPI inicializa cada um de forma isolada
— falha em um não impede a inicialização dos outros.
"""
from enum import Enum
from typing import Optional


class ComponentStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    INITIALIZING = "initializing"


class ComponentHealth:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = ComponentStatus.INITIALIZING
        self.error: Optional[str] = None

    def set_ready(self) -> None:
        self.status = ComponentStatus.READY
        self.error = None

    def set_failed(self, error: str) -> None:
        self.status = ComponentStatus.FAILED
        self.error = error

    def set_degraded(self, error: str) -> None:
        self.status = ComponentStatus.DEGRADED
        self.error = error

    @property
    def is_available(self) -> bool:
        return self.status in (ComponentStatus.READY, ComponentStatus.DEGRADED)

    def to_dict(self) -> dict:
        result: dict = {"status": self.status.value}
        if self.error:
            result["error"] = self.error
        return result


def overall_status(components: dict[str, ComponentHealth]) -> str:
    statuses = [h.status for h in components.values()]
    if all(s == ComponentStatus.READY for s in statuses):
        return "healthy"
    if any(s == ComponentStatus.FAILED for s in statuses):
        return "degraded"
    return "partial"
