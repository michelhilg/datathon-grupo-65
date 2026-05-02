"""Métricas Prometheus customizadas para a API de retenção de churn."""
import logging
import time
from typing import TYPE_CHECKING

from prometheus_client import Counter, Histogram
from prometheus_client.registry import CollectorRegistry
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Singletons no nível de módulo — evita CollectorRegistry duplicado em reloads
def _make_counter(name: str, doc: str, labels: list[str]) -> Counter:
    try:
        return Counter(name, doc, labels)
    except ValueError:
        from prometheus_client import REGISTRY  # noqa: PLC0415
        return REGISTRY._names_to_collectors.get(name)  # type: ignore[return-value]


def _make_histogram(name: str, doc: str, labels: list[str], buckets) -> Histogram:
    try:
        return Histogram(name, doc, labels, buckets=buckets)
    except ValueError:
        from prometheus_client import REGISTRY  # noqa: PLC0415
        return REGISTRY._names_to_collectors.get(name)  # type: ignore[return-value]


REQUEST_COUNT = _make_counter(
    "churn_api_requests_total",
    "Total de requisições à API por endpoint e status HTTP.",
    ["endpoint", "status_code"],
)

REQUEST_LATENCY = _make_histogram(
    "churn_api_request_latency_seconds",
    "Latência das requisições à API por endpoint (segundos).",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0],
)

CHURN_PROBABILITY_HISTOGRAM = _make_histogram(
    "churn_probability_distribution",
    "Distribuição das probabilidades de churn preditas.",
    [],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

TOOL_CALL_COUNTER = _make_counter(
    "agent_tool_calls_total",
    "Total de chamadas às tools do agente ReAct.",
    ["tool_name", "status"],
)


class _PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware Starlette que instrumenta REQUEST_COUNT e REQUEST_LATENCY."""

    async def dispatch(self, request: Request, call_next):
        endpoint = request.url.path
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        REQUEST_COUNT.labels(endpoint=endpoint, status_code=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)

        return response


def setup_prometheus_middleware(app: "FastAPI") -> None:
    """Registra o middleware Prometheus na aplicação FastAPI."""
    app.add_middleware(_PrometheusMiddleware)
    logger.info("Prometheus middleware registrado")
