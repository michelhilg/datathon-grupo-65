"""Envia 30 requisições variadas ao /predict para popular o DriftDetector."""
import json
import random
import sys
import urllib.request
from urllib.error import URLError

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["Fiber optic", "DSL", "No"]
SECURITY = ["Yes", "No", "No internet service"]
PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
YES_NO = ["Yes", "No"]

random.seed(42)


def make_customer(tenure: int, monthly: float) -> dict:
    total = round(tenure * monthly + random.uniform(-10, 10), 2)
    return {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": str(max(0.0, total)),
        "Contract": random.choice(CONTRACTS),
        "InternetService": random.choice(INTERNET),
        "OnlineSecurity": random.choice(SECURITY),
        "TechSupport": random.choice(SECURITY),
        "PaymentMethod": random.choice(PAYMENT),
        "PaperlessBilling": random.choice(YES_NO),
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.randint(0, 1),
        "Partner": random.choice(YES_NO),
        "Dependents": random.choice(YES_NO),
        "PhoneService": random.choice(YES_NO),
        "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
        "OnlineBackup": random.choice(SECURITY),
        "DeviceProtection": random.choice(SECURITY),
        "StreamingTV": random.choice(SECURITY),
        "StreamingMovies": random.choice(SECURITY),
    }


def predict(customer: dict) -> dict:
    payload = json.dumps({"customer_features": customer}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# Gera combinações variadas de tenure e MonthlyCharges
samples = [
    (tenure, monthly)
    for tenure in [1, 2, 5, 10, 15, 20, 24, 30, 36, 48]
    for monthly in [20.0, 45.5, 75.0]
][:30]

print(f"Enviando {len(samples)} requisições para {BASE_URL}/predict ...\n")

ok = err = 0
for i, (tenure, monthly) in enumerate(samples, 1):
    customer = make_customer(tenure, monthly)
    try:
        result = predict(customer)
        prob = result.get("churn_probability", "?")
        print(f"  [{i:02d}] tenure={tenure:2d}  monthly={monthly:5.1f}  churn_prob={prob:.3f}")
        ok += 1
    except URLError as e:
        print(f"  [{i:02d}] ERRO: {e}")
        err += 1

print(f"\nConcluído: {ok} OK | {err} erros")
if ok >= 30:
    print("\nMinimo de 30 amostras atingido. Rode agora:")
    print(f"  curl -X POST {BASE_URL}/drift-report | python3 -m json.tool")
