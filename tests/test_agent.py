"""Testes das tools do agente e do pipeline RAG — dependências externas mockadas."""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Modelo sklearn mockado com predict_proba e feature_importances_."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.22, 0.78]])
    model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02])
    return model


CUSTOMER_RAW = {
    "tenure": 2,
    "MonthlyCharges": 85.0,
    "TotalCharges": "170.0",
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
}


# ---------------------------------------------------------------------------
# churn_predictor
# ---------------------------------------------------------------------------

def test_churn_predictor_retorna_json_valido(mock_model):
    with patch("src.agent.tools._get_model", return_value=mock_model):
        from src.agent.tools import churn_predictor
        result = churn_predictor.invoke(json.dumps(CUSTOMER_RAW))

    data = json.loads(result)
    assert "churn_probability" in data
    assert "prediction" in data
    assert "risk_level" in data


def test_churn_predictor_alto_risco(mock_model):
    with patch("src.agent.tools._get_model", return_value=mock_model):
        from src.agent.tools import churn_predictor
        result = churn_predictor.invoke(json.dumps(CUSTOMER_RAW))

    data = json.loads(result)
    assert data["prediction"] == "Churn"
    assert data["risk_level"] == "Alto"
    assert data["churn_probability"] == pytest.approx(0.78, abs=0.01)


def test_churn_predictor_baixo_risco():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.85, 0.15]])
    with patch("src.agent.tools._get_model", return_value=model):
        from src.agent.tools import churn_predictor
        result = churn_predictor.invoke(json.dumps(CUSTOMER_RAW))

    data = json.loads(result)
    assert data["prediction"] == "No Churn"
    assert data["risk_level"] == "Baixo"


def test_churn_predictor_json_invalido():
    from src.agent.tools import churn_predictor
    result = churn_predictor.invoke("nao e json")
    data = json.loads(result)
    assert "error" in data


# ---------------------------------------------------------------------------
# feature_importance
# ---------------------------------------------------------------------------

def test_feature_importance_retorna_top5(mock_model):
    with patch("src.agent.tools._get_model", return_value=mock_model):
        from src.agent.tools import feature_importance
        result = feature_importance.invoke(json.dumps(CUSTOMER_RAW))

    data = json.loads(result)
    assert "top_5_risk_factors" in data
    assert len(data["top_5_risk_factors"]) == 5


def test_feature_importance_ordenado_por_importancia(mock_model):
    with patch("src.agent.tools._get_model", return_value=mock_model):
        from src.agent.tools import feature_importance
        result = feature_importance.invoke(json.dumps(CUSTOMER_RAW))

    factors = json.loads(result)["top_5_risk_factors"]
    importances = [f["importance"] for f in factors]
    assert importances == sorted(importances, reverse=True)


def test_feature_importance_modelo_sem_importances():
    model = MagicMock(spec=[])
    with patch("src.agent.tools._get_model", return_value=model):
        from src.agent.tools import feature_importance
        result = feature_importance.invoke(json.dumps(CUSTOMER_RAW))

    data = json.loads(result)
    assert "error" in data


# ---------------------------------------------------------------------------
# retention_knowledge (via make_retention_knowledge_tool)
# ---------------------------------------------------------------------------

def test_retention_knowledge_retorna_resultado():
    mock_collection = MagicMock()
    with patch("src.agent.rag_pipeline.retrieve", return_value="Estratégia A\n\nEstratégia B"):
        from src.agent.tools import make_retention_knowledge_tool
        tool = make_retention_knowledge_tool(mock_collection)
        result = tool.invoke("como reter clientes com contrato mensal?")

    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# RAG pipeline — retrieve
# ---------------------------------------------------------------------------

def test_retrieve_formata_resultado_com_fonte():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Clientes com contrato mensal têm maior churn."]],
        "metadatas": [[{"source": "churn_patterns.md"}]],
    }

    from src.agent.rag_pipeline import retrieve
    result = retrieve(query="churn mensal", collection=mock_collection, top_k=1)

    assert "churn_patterns.md" in result
    assert "Clientes com contrato mensal" in result


def test_retrieve_multiplos_chunks():
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Chunk A", "Chunk B"]],
        "metadatas": [[{"source": "file1.md"}, {"source": "file2.md"}]],
    }

    from src.agent.rag_pipeline import retrieve
    result = retrieve(query="teste", collection=mock_collection, top_k=2)

    assert "---" in result
    assert "Chunk A" in result
    assert "Chunk B" in result


# ---------------------------------------------------------------------------
# RAG pipeline — _chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_divide_corretamente():
    from src.agent.rag_pipeline import _chunk_text
    text = " ".join(["palavra"] * 600)
    chunks = _chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_texto_curto_retorna_um_chunk():
    from src.agent.rag_pipeline import _chunk_text
    chunks = _chunk_text("texto curto aqui", chunk_size=512, overlap=64)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# react_agent — analyze_customer
# ---------------------------------------------------------------------------

def test_analyze_customer_usa_question_padrao():
    mock_message = MagicMock()
    mock_message.content = "Análise de churn completa."
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": [mock_message]}

    from src.agent.react_agent import analyze_customer
    result = analyze_customer(agent=mock_agent, customer_json=json.dumps(CUSTOMER_RAW))

    assert result == "Análise de churn completa."
    call_args = mock_agent.invoke.call_args[0][0]
    assert "messages" in call_args


def test_analyze_customer_usa_question_customizada():
    mock_message = MagicMock()
    mock_message.content = "Resposta customizada."
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": [mock_message]}

    from src.agent.react_agent import analyze_customer
    result = analyze_customer(
        agent=mock_agent,
        customer_json=json.dumps(CUSTOMER_RAW),
        question="Qual o risco?",
    )

    assert result == "Resposta customizada."
    call_args = mock_agent.invoke.call_args[0][0]
    assert call_args["messages"][0].content == "Qual o risco?"
