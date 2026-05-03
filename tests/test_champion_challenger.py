"""Testes para scripts/champion_challenger.py"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.champion_challenger import (
    _decide,
    _load_champion_metrics,
    _write_github_output,
    _write_updated_metrics,
)


# ---------------------------------------------------------------------------
# _decide — lógica de decisão pura (sem I/O)
# ---------------------------------------------------------------------------


def test_decide_promotes_when_delta_exceeds_threshold():
    assert _decide(0.80, 0.81, 0.005) == "promote"


def test_decide_promotes_when_delta_equals_threshold():
    assert _decide(0.80, 0.805, 0.005) == "promote"


def test_decide_skips_when_delta_below_threshold():
    assert _decide(0.80, 0.803, 0.005) == "skip"


def test_decide_skips_when_challenger_is_worse():
    assert _decide(0.80, 0.75, 0.005) == "skip"


# ---------------------------------------------------------------------------
# _load_champion_metrics — leitura de evaluation/model_metrics.json
# ---------------------------------------------------------------------------


def test_load_champion_returns_none_when_file_missing(tmp_path):
    result = _load_champion_metrics(base_path=tmp_path)
    assert result is None


def test_load_champion_returns_none_when_not_validated(tmp_path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "model_metrics.json").write_text(
        json.dumps({"auc": 0.80, "validation_passed": False})
    )
    result = _load_champion_metrics(base_path=tmp_path)
    assert result is None


def test_load_champion_returns_metrics_when_valid(tmp_path):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    champion_data = {
        "run_id": "abc123",
        "model_name": "Random_Forest",
        "auc": 0.82,
        "f1": 0.65,
        "validation_passed": True,
    }
    (eval_dir / "model_metrics.json").write_text(json.dumps(champion_data))
    result = _load_champion_metrics(base_path=tmp_path)
    assert result is not None
    assert result["auc"] == 0.82
    assert result["run_id"] == "abc123"


# ---------------------------------------------------------------------------
# _write_github_output — escreve no arquivo GITHUB_OUTPUT
# ---------------------------------------------------------------------------


def test_write_github_output_writes_correctly(tmp_path, monkeypatch):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    _write_github_output(decision="promote", delta_auc="0.01", challenger_run_id="xyz")
    content = output_file.read_text()
    assert "decision=promote" in content
    assert "delta_auc=0.01" in content
    assert "challenger_run_id=xyz" in content


def test_write_github_output_is_noop_when_env_not_set(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    _write_github_output(decision="skip")  # não deve levantar exceção


# ---------------------------------------------------------------------------
# _write_updated_metrics — atualiza evaluation/model_metrics.json
# ---------------------------------------------------------------------------


def test_write_updated_metrics_creates_valid_file(tmp_path):
    params = {
        "model": {"min_auc_threshold": 0.75},
        "champion_challenger": {"experiment_name": "Telco_Test_Retraining"},
    }
    challenger = {
        "run_id": "chal-999",
        "name": "Challenger_RF",
        "auc": 0.83,
        "f1": 0.67,
        "precision": 0.72,
        "recall": 0.63,
    }
    _write_updated_metrics(challenger, params, base_path=tmp_path)

    metrics_path = tmp_path / "evaluation" / "model_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["run_id"] == "chal-999"
    assert metrics["auc"] == 0.83
    assert metrics["validation_passed"] is True
    assert metrics["promoted_by"] == "champion_challenger"


def test_write_updated_metrics_marks_invalid_below_threshold(tmp_path):
    params = {
        "model": {"min_auc_threshold": 0.90},  # threshold alto
        "champion_challenger": {"experiment_name": "Test"},
    }
    challenger = {
        "run_id": "chal-low",
        "name": "Challenger_LR",
        "auc": 0.77,
        "f1": 0.55,
        "precision": 0.60,
        "recall": 0.50,
    }
    _write_updated_metrics(challenger, params, base_path=tmp_path)
    metrics = json.loads((tmp_path / "evaluation" / "model_metrics.json").read_text())
    assert metrics["validation_passed"] is False
