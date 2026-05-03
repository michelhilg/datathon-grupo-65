"""Pacote de monitoramento — métricas Prometheus, telemetria Langfuse e drift detection."""

from src.monitoring.drift import DriftDetector
from src.monitoring.metrics import (
    CHURN_PROBABILITY_HISTOGRAM,
    DRIFT_PSI_GAUGE,
    DRIFT_RETRAIN_GAUGE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SECURITY_BLOCK_COUNTER,
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
    "SECURITY_BLOCK_COUNTER",
    "DRIFT_PSI_GAUGE",
    "DRIFT_RETRAIN_GAUGE",
    "get_langfuse_handler",
    "ContextAccumulatorHandler",
    "DriftDetector",
]
