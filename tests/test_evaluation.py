"""Testes dos scripts de avaliação — golden set, RAGAS helpers e LLM judge."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

GOLDEN_SET_PATH = Path("data/golden_set/golden_set.json")


# ---------------------------------------------------------------------------
# golden set
# ---------------------------------------------------------------------------

def test_golden_set_file_exists():
    assert GOLDEN_SET_PATH.exists(), "golden_set.json não encontrado"


def test_golden_set_loads_and_has_minimum_pairs():
    data = json.loads(GOLDEN_SET_PATH.read_text())
    assert "pairs" in data
    assert len(data["pairs"]) >= 20, f"Esperado ≥20 pares, encontrado {len(data['pairs'])}"


def test_golden_set_schema_valid():
    data = json.loads(GOLDEN_SET_PATH.read_text())
    required_keys = {"id", "profile_type", "customer_features", "question", "ground_truth",
                     "expected_risk_level", "expected_churn_range", "tags"}
    for pair in data["pairs"]:
        missing = required_keys - set(pair.keys())
        assert not missing, f"Par {pair.get('id')} faltando campos: {missing}"


def test_golden_set_has_all_profile_types():
    data = json.loads(GOLDEN_SET_PATH.read_text())
    profiles = {p["profile_type"] for p in data["pairs"]}
    assert "high_risk" in profiles
    assert "medium_risk" in profiles
    assert "low_risk" in profiles
    assert "edge_case" in profiles


def test_golden_set_churn_ranges_valid():
    data = json.loads(GOLDEN_SET_PATH.read_text())
    for pair in data["pairs"]:
        lo, hi = pair["expected_churn_range"]
        assert 0.0 <= lo <= 1.0, f"Par {pair['id']}: lower bound inválido"
        assert 0.0 <= hi <= 1.0, f"Par {pair['id']}: upper bound inválido"
        assert lo <= hi, f"Par {pair['id']}: range invertido"


def test_golden_set_risk_levels_valid():
    data = json.loads(GOLDEN_SET_PATH.read_text())
    valid = {"Alto", "Médio", "Baixo"}
    for pair in data["pairs"]:
        assert pair["expected_risk_level"] in valid, \
            f"Par {pair['id']}: risk_level inválido '{pair['expected_risk_level']}'"


# ---------------------------------------------------------------------------
# ContextAccumulatorHandler (reuso do módulo de monitoring)
# ---------------------------------------------------------------------------

def test_context_capture_handler_splits_on_separator():
    from src.monitoring.telemetry import ContextAccumulatorHandler
    handler = ContextAccumulatorHandler()
    handler.on_tool_end(
        "[churn_patterns.md]\nEstatísticas de churn\n\n---\n\n[retention_strategies.md]\nEstratégias de retenção",
        run_id=uuid4(),
    )
    assert len(handler.captured_contexts) == 2


def test_context_capture_empty_output():
    from src.monitoring.telemetry import ContextAccumulatorHandler
    handler = ContextAccumulatorHandler()
    handler.on_tool_end("", run_id=uuid4())
    assert handler.captured_contexts == []


# ---------------------------------------------------------------------------
# ragas_eval.py helpers
# ---------------------------------------------------------------------------

def test_build_ragas_dataset_structure():
    """build_ragas_dataset deve retornar lista com campos obrigatórios."""
    from evaluation.ragas_eval import build_ragas_dataset

    mock_answer = "O cliente tem 85% de probabilidade de churn."
    mock_contexts = ["[churn_patterns.md]\nFiber optic tem 41.9% de churn."]

    pairs = [json.loads(Path("data/golden_set/golden_set.json").read_text())["pairs"][0]]

    with patch("evaluation.ragas_eval._run_agent_direct", return_value=(mock_answer, mock_contexts)):
        rows = build_ragas_dataset(pairs, mode="direct", api_url="", limit=None)

    assert len(rows) == 1
    row = rows[0]
    assert "question" in row
    assert "answer" in row
    assert "contexts" in row
    assert "ground_truth" in row
    assert row["answer"] == mock_answer
    assert row["contexts"] == mock_contexts


def test_build_ragas_dataset_uses_placeholder_when_no_contexts():
    from evaluation.ragas_eval import build_ragas_dataset

    pairs = [json.loads(Path("data/golden_set/golden_set.json").read_text())["pairs"][0]]

    with patch("evaluation.ragas_eval._run_agent_direct", return_value=("resposta", [])):
        rows = build_ragas_dataset(pairs, mode="direct", api_url="", limit=None)

    assert rows[0]["contexts"] == ["[contexto não disponível]"]


# ---------------------------------------------------------------------------
# llm_judge.py — parse_judge_response
# ---------------------------------------------------------------------------

def test_parse_judge_response_valid_json():
    from evaluation.llm_judge import parse_judge_response

    raw = json.dumps({
        "fidelidade_aos_dados": 8,
        "relevancia_estrategica": 7,
        "valor_acao_retencao": 9,
        "clareza_estrutura": 8,
        "justificativa": "Resposta precisa e bem estruturada.",
    })
    result = parse_judge_response(raw)
    assert result["fidelidade_aos_dados"] == 8
    assert result["valor_acao_retencao"] == 9
    assert "_parse_error" not in result


def test_parse_judge_response_with_markdown_wrapper():
    from evaluation.llm_judge import parse_judge_response

    raw = '```json\n{"fidelidade_aos_dados": 7, "relevancia_estrategica": 6, "valor_acao_retencao": 8, "clareza_estrutura": 9, "justificativa": "ok"}\n```'
    result = parse_judge_response(raw)
    assert result["fidelidade_aos_dados"] == 7
    assert "_parse_error" not in result


def test_parse_judge_response_malformed_json():
    from evaluation.llm_judge import parse_judge_response

    result = parse_judge_response("isso não é json válido {{{")
    assert result["_parse_error"] is True
    assert result["fidelidade_aos_dados"] == 0


def test_parse_judge_response_missing_field():
    from evaluation.llm_judge import parse_judge_response

    raw = json.dumps({"fidelidade_aos_dados": 8, "relevancia_estrategica": 7})
    result = parse_judge_response(raw)
    assert result["_parse_error"] is True


def test_customer_summary_format():
    from evaluation.llm_judge import _customer_summary

    features = {
        "tenure": 12,
        "MonthlyCharges": 75.50,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "SeniorCitizen": 0,
    }
    summary = _customer_summary(features)
    assert "12" in summary
    assert "75.5" in summary
    assert "Month-to-month" in summary
