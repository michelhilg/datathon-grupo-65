"""Testes do endpoint FastAPI — LLM mockado, sem chamadas reais à OpenAI."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


CUSTOMER_FIXTURE = {
    "tenure": 2,
    "MonthlyCharges": 75.50,
    "TotalCharges": "151.00",
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


@pytest.fixture
def client():
    """TestClient com lifespan mockado — sem RAG real nem MLflow."""
    mock_message = MagicMock()
    mock_message.content = (
        "Risco de churn: Alto (78%). "
        "Principais fatores: contrato mensal, fibra óptica, tenure baixo. "
        "Recomendação: oferecer upgrade para contrato anual com 20% de desconto."
    )
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"messages": [mock_message]}

    with patch("src.serving.app.build_index") as mock_index, \
         patch("src.serving.app.build_tools") as mock_tools, \
         patch("src.serving.app.create_churn_agent") as mock_create:

        mock_index.return_value = MagicMock()
        mock_tools.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_create.return_value = mock_agent

        from src.serving.app import app
        with TestClient(app) as c:
            yield c


def test_health_retorna_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "partial", "degraded")
    assert data["capabilities"]["predict"] is True


def test_analyze_retorna_200(client):
    response = client.post("/analyze", json={"customer_features": CUSTOMER_FIXTURE})
    assert response.status_code == 200


def test_analyze_resposta_tem_campo_analysis(client):
    response = client.post("/analyze", json={"customer_features": CUSTOMER_FIXTURE})
    data = response.json()
    assert "analysis" in data
    assert isinstance(data["analysis"], str)
    assert len(data["analysis"]) > 0


def test_analyze_com_pergunta_customizada(client):
    payload = {
        "customer_features": CUSTOMER_FIXTURE,
        "question": "Qual a probabilidade de churn deste cliente?",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200


def test_analyze_payload_invalido_retorna_422(client):
    response = client.post("/analyze", json={"customer_features": "não é dict"})
    assert response.status_code == 422


def test_docs_disponiveis(client):
    """Swagger UI deve estar acessível."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_metrics_endpoint_retorna_prometheus(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert b"churn_api_requests_total" in response.content


def test_analyze_include_contexts_retorna_lista(client):
    payload = {"customer_features": CUSTOMER_FIXTURE, "include_contexts": True}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "contexts" in data
    # contexts pode ser lista vazia ou lista de strings (agente mockado não dispara RAG real)
    assert data["contexts"] is not None
    assert isinstance(data["contexts"], list)


def test_analyze_sem_include_contexts_omite_campo(client):
    payload = {"customer_features": CUSTOMER_FIXTURE, "include_contexts": False}
    response = client.post("/analyze", json=payload)
    data = response.json()
    assert data.get("contexts") is None


def test_drift_report_retorna_insufficient_data(client):
    response = client.post("/drift-report")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("insufficient_data", "ok", "error")
