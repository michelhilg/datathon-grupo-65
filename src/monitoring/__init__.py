"""Pacote de monitoramento — métricas Prometheus, telemetria Langfuse e drift detection."""

from src.monitoring.drift import DriftDetector
from src.monitoring.metrics import (
    CHURN_PROBABILITY_HISTOGRAM,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TOOL_CALL_COUNTER,
    setup_prometheus_middleware,
)
from src.monitoring.telemetry import ContextAccumulatorHandler, get_langfuse_handler

__all__ = [
    "setup_prometheus_middleware",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "CHURN_PROBABILITY_HISTOGRAM",
    "TOOL_CALL_COUNTER",
    "get_langfuse_handler",
    "ContextAccumulatorHandler",
    "DriftDetector",
]
