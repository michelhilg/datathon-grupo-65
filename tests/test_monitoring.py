"""Testes do pacote src/monitoring — métricas, telemetria e drift."""
import os
import sys
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

def test_request_count_labels():
    from src.monitoring.metrics import REQUEST_COUNT
    REQUEST_COUNT.labels(endpoint="/test", status_code="200").inc()
    # Se não lançar exceção, os labels estão corretos


def test_request_latency_observe():
    from src.monitoring.metrics import REQUEST_LATENCY
    REQUEST_LATENCY.labels(endpoint="/test").observe(0.5)


def test_churn_histogram_observe():
    from src.monitoring.metrics import CHURN_PROBABILITY_HISTOGRAM
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        CHURN_PROBABILITY_HISTOGRAM.observe(v)


def test_tool_call_counter_increments():
    from src.monitoring.metrics import TOOL_CALL_COUNTER
    TOOL_CALL_COUNTER.labels(tool_name="churn_predictor", status="success").inc()
    TOOL_CALL_COUNTER.labels(tool_name="analyze", status="error").inc()


def test_setup_prometheus_middleware_registers(monkeypatch):
    from fastapi import FastAPI
    from src.monitoring.metrics import setup_prometheus_middleware
    app = FastAPI()
    setup_prometheus_middleware(app)
    middleware_types = [type(m).__name__ for m in app.user_middleware]
    assert any("Prometheus" in t or "Middleware" in t for t in middleware_types)


# ---------------------------------------------------------------------------
# telemetry.py — ContextAccumulatorHandler
# ---------------------------------------------------------------------------

def test_context_accumulator_captures_chunks():
    from src.monitoring.telemetry import ContextAccumulatorHandler
    handler = ContextAccumulatorHandler()
    output = "[churn_patterns.md]\nChunk 1\n\n---\n\n[retention_strategies.md]\nChunk 2"
    handler.on_tool_end(output, run_id=uuid4())
    assert len(handler.captured_contexts) == 2
    assert "Chunk 1" in handler.captured_contexts[0]
    assert "Chunk 2" in handler.captured_contexts[1]


def test_context_accumulator_ignores_other_tools():
    from src.monitoring.telemetry import ContextAccumulatorHandler
    handler = ContextAccumulatorHandler()
    handler.on_tool_end('{"churn_probability": 0.85}', run_id=uuid4())
    assert handler.captured_contexts == []


def test_context_accumulator_accumulates_multiple_calls():
    from src.monitoring.telemetry import ContextAccumulatorHandler
    handler = ContextAccumulatorHandler()
    handler.on_tool_end("chunk A\n\n---\n\nchunk B", run_id=uuid4())
    handler.on_tool_end("chunk C\n\n---\n\nchunk D", run_id=uuid4())
    assert len(handler.captured_contexts) == 4


# ---------------------------------------------------------------------------
# telemetry.py — get_langfuse_handler
# ---------------------------------------------------------------------------

def test_get_langfuse_handler_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from src.monitoring.telemetry import get_langfuse_handler
    handler = get_langfuse_handler()
    assert handler is None


def test_get_langfuse_handler_disabled_without_package(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    with patch.dict(sys.modules, {"langfuse": None}):
        from src.monitoring.telemetry import get_langfuse_handler
        handler = get_langfuse_handler()
        assert handler is None


def test_get_langfuse_handler_returns_new_instance(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    
    mock_langfuse_module = MagicMock()
    mock_langchain_module = MagicMock()
    
    class DummyCallbackHandler:
        pass

    mock_langchain_module.CallbackHandler = DummyCallbackHandler

    with patch.dict(sys.modules, {
        "langfuse": mock_langfuse_module, 
        "langfuse.langchain": mock_langchain_module
    }):
        from src.monitoring.telemetry import get_langfuse_handler
        h1 = get_langfuse_handler()
        h2 = get_langfuse_handler()
        assert h1 is not None
        assert h1 is not h2
        assert isinstance(h1, DummyCallbackHandler)


# ---------------------------------------------------------------------------
# drift.py — DriftDetector
# ---------------------------------------------------------------------------

def test_drift_detector_record_accumulates():
    from src.monitoring.drift import DriftDetector
    detector = DriftDetector(window_size=10)
    features = {"tenure": 5, "MonthlyCharges": 50.0, "TotalCharges": "250.00"}
    detector.record(features)
    detector.record(features)
    assert len(detector._window) == 2


def test_drift_detector_record_handles_empty_total_charges():
    from src.monitoring.drift import DriftDetector
    detector = DriftDetector()
    detector.record({"tenure": 0, "MonthlyCharges": 70.0, "TotalCharges": ""})
    assert detector._window[0]["TotalCharges"] == 0.0


def test_drift_detector_compute_psi_stable():
    from src.monitoring.drift import DriftDetector
    detector = DriftDetector()
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    psi = detector._compute_psi(series, series.copy())
    assert psi < 0.05


def test_drift_detector_compute_psi_drifted():
    from src.monitoring.drift import DriftDetector
    detector = DriftDetector()
    ref = pd.Series([1.0] * 100)
    curr = pd.Series([10.0] * 100)
    psi = detector._compute_psi(ref, curr)
    assert psi > DriftDetector.PSI_RETRAIN


def test_drift_detector_run_report_insufficient_data():
    from src.monitoring.drift import DriftDetector
    detector = DriftDetector()
    for _ in range(5):
        detector.record({"tenure": 5, "MonthlyCharges": 50.0, "TotalCharges": "250.00"})
    result = detector.run_report()
    assert result["status"] == "insufficient_data"
    assert result["n_current_samples"] == 5


def test_drift_detector_run_report_missing_reference(tmp_path):
    from src.monitoring.drift import DriftDetector
    # Passa reference_path inexistente via construtor (não mais via class attribute)
    detector = DriftDetector(reference_path=tmp_path / "nonexistent.csv")
    for _ in range(35):
        detector.record({"tenure": 5, "MonthlyCharges": 50.0, "TotalCharges": "250.00"})
    result = detector.run_report()
    assert result["status"] == "error"
