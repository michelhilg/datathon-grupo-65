"""Fixtures compartilhados para testes — nunca dados reais."""
import pandas as pd
import pytest

from tests.fixtures.synthetic_data import generate_synthetic_telco


@pytest.fixture
def telco_raw() -> pd.DataFrame:
    """Dados sintéticos no formato bruto do Telco Customer Churn.

    16 linhas com mix de todos os valores possíveis, incluindo:
    - tenure=0 com TotalCharges em branco (caso especial de convert_types)
    - Todos os valores de Contract, PaymentMethod e InternetService
    - 8 churners e 8 não-churners (balanceado para estratificação no treino)
    """
    return pd.DataFrame({
        "customerID": [f"CUST-{i:04d}" for i in range(16)],
        "gender": [
            "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female",
        ],
        "SeniorCitizen": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "Partner": [
            "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No",
            "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No",
        ],
        "Dependents": [
            "No", "No", "Yes", "No", "Yes", "No", "No", "Yes",
            "No", "No", "Yes", "No", "Yes", "No", "No", "Yes",
        ],
        "tenure": [0, 1, 6, 12, 24, 36, 48, 60, 3, 18, 30, 42, 54, 8, 15, 72],
        "PhoneService": [
            "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes",
            "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes",
        ],
        "MultipleLines": [
            "No", "No phone service", "Yes", "No", "Yes", "No phone service",
            "No", "Yes", "No", "No phone service", "Yes", "No",
            "Yes", "No phone service", "No", "Yes",
        ],
        "InternetService": [
            "DSL", "Fiber optic", "No", "DSL", "Fiber optic", "DSL",
            "No", "Fiber optic", "DSL", "Fiber optic", "No", "DSL",
            "Fiber optic", "No", "DSL", "Fiber optic",
        ],
        "OnlineSecurity": [
            "Yes", "No", "No internet service", "Yes", "No", "Yes",
            "No internet service", "No", "Yes", "No", "No internet service",
            "Yes", "No", "No internet service", "Yes", "No",
        ],
        "OnlineBackup": [
            "No", "Yes", "No internet service", "Yes", "No", "Yes",
            "No internet service", "No", "Yes", "No", "No internet service",
            "Yes", "No", "No internet service", "No", "Yes",
        ],
        "DeviceProtection": [
            "Yes", "No", "No internet service", "No", "Yes", "No",
            "No internet service", "Yes", "No", "Yes", "No internet service",
            "No", "Yes", "No internet service", "Yes", "No",
        ],
        "TechSupport": [
            "No", "No", "No internet service", "Yes", "No", "Yes",
            "No internet service", "No", "Yes", "No", "No internet service",
            "Yes", "No", "No internet service", "No", "Yes",
        ],
        "StreamingTV": [
            "No", "Yes", "No internet service", "No", "Yes", "No",
            "No internet service", "Yes", "No", "Yes", "No internet service",
            "No", "Yes", "No internet service", "No", "Yes",
        ],
        "StreamingMovies": [
            "Yes", "No", "No internet service", "Yes", "No", "Yes",
            "No internet service", "No", "Yes", "No", "No internet service",
            "Yes", "No", "No internet service", "Yes", "No",
        ],
        "Contract": [
            "Month-to-month", "One year", "Two year", "Month-to-month", "One year",
            "Two year", "Month-to-month", "Two year", "Month-to-month", "One year",
            "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "One year",
        ],
        "PaperlessBilling": [
            "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No",
            "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No",
        ],
        "PaymentMethod": [
            "Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)", "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)", "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ],
        "MonthlyCharges": [
            29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 20.65, 89.10,
            29.75, 74.45, 56.05, 49.50, 65.60, 24.10, 38.95, 109.70,
        ],
        # TotalCharges vem como string no CSV; tenure=0 tem espaço em branco
        "TotalCharges": [
            " ", "56.95", "323.10", "507.60", "1685.20", "3585.40",
            "990.70", "5353.40", "89.25", "1340.10", "1680.90",
            "2080.10", "3540.90", "192.80", "584.25", "7905.00",
        ],
        "Churn": [
            "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes",
            "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes",
        ],
    })


@pytest.fixture
def telco_synthetic_large() -> pd.DataFrame:
    """Dataset sintético de 500 clientes para testes que precisam de volume.

    Gerado deterministicamente via seed=42 — resultados estáveis entre runs.
    Nunca lê data/raw/ — zero dependência de dados reais em testes.
    """
    return generate_synthetic_telco(n_rows=500, seed=42)
