"""Benchmark do agente ReAct — 3 configurações de RAG e temperatura.

Executa o agente com as configurações definidas em configs/model_config.yaml
e salva resultados em evaluation/benchmark_results.json.

Uso:
    uv run python evaluation/benchmark.py
"""
import json
import logging
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_config.yaml"
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"

# Casos de teste representativos
TEST_CASES = [
    {
        "id": "high_risk_customer",
        "description": "Cliente de alto risco (month-to-month, fiber, tenure baixo)",
        "customer_features": {
            "tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": "170.0",
            "Contract": "Month-to-month", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "TechSupport": "No",
            "PaymentMethod": "Electronic check", "PaperlessBilling": "Yes",
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "StreamingTV": "No", "StreamingMovies": "No",
        },
    },
    {
        "id": "low_risk_customer",
        "description": "Cliente de baixo risco (contrato anual, DSL, tenure alto)",
        "customer_features": {
            "tenure": 48, "MonthlyCharges": 45.0, "TotalCharges": "2160.0",
            "Contract": "Two year", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "TechSupport": "Yes",
            "PaymentMethod": "Bank transfer (automatic)", "PaperlessBilling": "No",
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "PhoneService": "Yes", "MultipleLines": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
        },
    },
    {
        "id": "medium_risk_customer",
        "description": "Cliente de risco médio (contrato anual, fiber, sem suporte)",
        "customer_features": {
            "tenure": 14, "MonthlyCharges": 70.0, "TotalCharges": "980.0",
            "Contract": "One year", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "TechSupport": "No",
            "PaymentMethod": "Credit card (automatic)", "PaperlessBilling": "Yes",
            "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        },
    },
]


def run_benchmark():
    """Executa benchmark com as 3 configurações e salva resultados."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from src.agent.rag_pipeline import build_index
    from src.agent.tools import build_tools
    from src.agent.react_agent import create_churn_agent, analyze_customer

    logger.info("Inicializando knowledge base...")
    base_collection = build_index()

    benchmark_configs = config["benchmark"]["configs"]
    results = {"configs": [], "summary": {}}

    for bench_config in benchmark_configs:
        config_name = bench_config["name"]
        top_k = bench_config["top_k"]
        temperature = bench_config["temperature"]

        logger.info("Executando configuração: %s (top_k=%d, temp=%.1f)", config_name, top_k, temperature)

        collection = build_index() if top_k != config["rag"]["top_k"] else base_collection
        tools = build_tools(collection)
        agent = create_churn_agent(
            tools=tools,
            model_name=config["llm"]["model"],
            temperature=temperature,
        )

        config_results = {
            "config": {"name": config_name, "top_k": top_k, "temperature": temperature,
                       "model": config["llm"]["model"]},
            "test_cases": [],
        }

        for test_case in TEST_CASES:
            import json as _json
            customer_json = _json.dumps(test_case["customer_features"])
            start = time.time()

            try:
                response = analyze_customer(agent=agent, customer_json=customer_json)
                latency_ms = round((time.time() - start) * 1000, 1)

                config_results["test_cases"].append({
                    "id": test_case["id"],
                    "description": test_case["description"],
                    "latency_ms": latency_ms,
                    "response_length": len(response),
                    "response_preview": response[:200],
                    "status": "ok",
                })
                logger.info("  %s: %dms, %d chars", test_case["id"], latency_ms, len(response))

            except Exception as exc:
                config_results["test_cases"].append({
                    "id": test_case["id"],
                    "status": "error",
                    "error": str(exc),
                })
                logger.error("  %s: ERRO — %s", test_case["id"], exc)

        results["configs"].append(config_results)

    # Resumo comparativo
    for cfg in results["configs"]:
        ok_cases = [tc for tc in cfg["test_cases"] if tc["status"] == "ok"]
        if ok_cases:
            avg_latency = sum(tc["latency_ms"] for tc in ok_cases) / len(ok_cases)
            avg_length = sum(tc["response_length"] for tc in ok_cases) / len(ok_cases)
            results["summary"][cfg["config"]["name"]] = {
                "avg_latency_ms": round(avg_latency, 1),
                "avg_response_length": round(avg_length, 1),
                "success_rate": f"{len(ok_cases)}/{len(cfg['test_cases'])}",
            }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Resultados salvos em %s", RESULTS_PATH)
    return results


if __name__ == "__main__":
    run_benchmark()
