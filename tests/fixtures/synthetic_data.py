"""Gerador de dados sintéticos no formato Telco Customer Churn.

Usado tanto pelo stage DVC generate_dev_data quanto pelas fixtures pytest.
Garante que testes nunca dependam do dataset real — mudanças de schema no CSV
de produção não quebram a suíte de testes.

Distribuições:
  - tenure       : uniforme [0, 72]
  - MonthlyCharges: uniforme [18.00, 120.00]
  - TotalCharges : calculado como monthly * tenure + ruído (string, como no CSV real)
  - Churn        : correlacionado com tenure curto + fibra óptica + contrato mensal
  - tenure=0     : TotalCharges em branco (replicando o caso especial do dataset real)
"""
import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
YES_NO = ["Yes", "No"]


def generate_synthetic_telco(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Gera DataFrame sintético no formato Telco Customer Churn.

    Args:
        n_rows: Número de clientes a gerar.
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame com as mesmas 21 colunas do dataset original.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, size=n_rows)
    monthly_charges = rng.uniform(18.0, 120.0, size=n_rows).round(2)
    total_charges_num = np.where(
        tenure == 0,
        monthly_charges,
        (monthly_charges * tenure + rng.normal(0, 10, size=n_rows)).clip(0).round(2),
    )
    # TotalCharges como string com espaço em branco para tenure=0 (mesmo comportamento do CSV real)
    total_charges_str = np.where(tenure == 0, " ", total_charges_num.astype(str))

    internet_service = rng.choice(INTERNET_SERVICES, size=n_rows)
    no_internet_mask = internet_service == "No"

    def _internet_dep(col_yes_no: np.ndarray) -> np.ndarray:
        return np.where(no_internet_mask, "No internet service", col_yes_no)

    phone_service = rng.choice(YES_NO, size=n_rows)
    no_phone_mask = phone_service == "No"
    multiple_lines = np.where(no_phone_mask, "No phone service", rng.choice(YES_NO, size=n_rows))

    contracts = rng.choice(CONTRACTS, size=n_rows)

    # Churn correlacionado com fatores reais de risco
    churn_prob = (
        0.10
        + 0.30 * (tenure < 12).astype(float)
        + 0.15 * (internet_service == "Fiber optic").astype(float)
        + 0.10 * (contracts == "Month-to-month").astype(float)
        - 0.15 * (contracts == "Two year").astype(float)
    ).clip(0.0, 0.95)
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    return pd.DataFrame({
        "customerID": [f"SYN-{i:05d}" for i in range(n_rows)],
        "gender": rng.choice(["Male", "Female"], size=n_rows),
        "SeniorCitizen": rng.integers(0, 2, size=n_rows),
        "Partner": rng.choice(YES_NO, size=n_rows),
        "Dependents": rng.choice(YES_NO, size=n_rows),
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": _internet_dep(rng.choice(YES_NO, size=n_rows)),
        "OnlineBackup": _internet_dep(rng.choice(YES_NO, size=n_rows)),
        "DeviceProtection": _internet_dep(rng.choice(YES_NO, size=n_rows)),
        "TechSupport": _internet_dep(rng.choice(YES_NO, size=n_rows)),
        "StreamingTV": _internet_dep(rng.choice(YES_NO, size=n_rows)),
        "StreamingMovies": _internet_dep(rng.choice(YES_NO, size=n_rows)),
        "Contract": contracts,
        "PaperlessBilling": rng.choice(YES_NO, size=n_rows),
        "PaymentMethod": rng.choice(PAYMENT_METHODS, size=n_rows),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_str,
        "Churn": churn,
    })
