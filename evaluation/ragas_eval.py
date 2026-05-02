"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias.

Uso:
    uv run python evaluation/ragas_eval.py [--mode direct|api] [--output PATH] [--limit N]

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of Retrieval
            Augmented Generation. https://arxiv.org/abs/2309.15217
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

# Adiciona a raiz do projeto ao sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/golden_set/golden_set.json")
DEFAULT_OUTPUT = Path("evaluation/ragas_results.json")


def _load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH) as f:
        data = json.load(f)
    pairs = data["pairs"]
    logger.info("Golden set carregado: %d pares", len(pairs))
    return pairs


def _run_agent_direct(pair: dict) -> tuple[str, list[str]]:
    """Executa o agente diretamente (sem servidor) e retorna (answer, contexts)."""
    from langchain_core.runnables.config import RunnableConfig  # noqa: PLC0415

    from src.agent.rag_pipeline import build_index  # noqa: PLC0415
    from src.agent.react_agent import analyze_customer  # noqa: PLC0415
    from src.agent.tools import build_tools  # noqa: PLC0415
    from src.monitoring.telemetry import ContextAccumulatorHandler  # noqa: PLC0415

    # Reutiliza índice/agente cacheados entre chamadas
    if not hasattr(_run_agent_direct, "_agent"):
        logger.info("Inicializando agente para avaliação direta...")
        collection = build_index()
        tools = build_tools(collection)

        from src.agent.react_agent import create_churn_agent  # noqa: PLC0415
        _run_agent_direct._agent = create_churn_agent(  # type: ignore[attr-defined]
            tools=tools,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.0,
        )

    ctx_handler = ContextAccumulatorHandler()
    config = RunnableConfig(callbacks=[ctx_handler])

    customer_json = json.dumps(pair["customer_features"], ensure_ascii=False)
    answer = analyze_customer(
        agent=_run_agent_direct._agent,  # type: ignore[attr-defined]
        customer_json=customer_json,
        question=pair.get("question"),
        config=config,
    )
    return answer, ctx_handler.captured_contexts


def _run_agent_api(pair: dict, api_url: str) -> tuple[str, list[str]]:
    """Chama a API /analyze com include_contexts=True."""
    import httpx  # noqa: PLC0415

    payload = {
        "customer_features": pair["customer_features"],
        "question": pair.get("question"),
        "include_contexts": True,
    }
    response = httpx.post(f"{api_url}/analyze", json=payload, timeout=60.0)
    response.raise_for_status()
    data = response.json()
    return data["analysis"], data.get("contexts") or []


def build_ragas_dataset(pairs: list[dict], mode: str, api_url: str, limit: int | None) -> list[dict]:
    """Executa o agente para cada par e monta lista de dicts para o Dataset RAGAS."""
    if limit:
        pairs = pairs[:limit]

    rows = []
    for i, pair in enumerate(pairs):
        logger.info("[%d/%d] Avaliando par %s...", i + 1, len(pairs), pair["id"])
        try:
            if mode == "direct":
                answer, contexts = _run_agent_direct(pair)
            else:
                answer, contexts = _run_agent_api(pair, api_url)

            if not contexts:
                logger.warning("Par %s: nenhum contexto capturado — usando placeholder.", pair["id"])
                contexts = ["[contexto não disponível]"]

            rows.append({
                "question": pair["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": pair["ground_truth"],
                "pair_id": pair["id"],
                "profile_type": pair["profile_type"],
            })
        except Exception as exc:
            logger.error("Erro no par %s: %s", pair["id"], exc)

    logger.info("Dataset RAGAS construído: %d linhas", len(rows))
    return rows


def run_ragas_evaluation(rows: list[dict]) -> dict:
    """Executa evaluate() do RAGAS e retorna métricas agregadas."""
    try:
        from datasets import Dataset  # noqa: PLC0415
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        from ragas import evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        logger.error("Dependência ausente para RAGAS: %s. Instale com: uv add --group dev ragas datasets", exc)
        sys.exit(1)

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)

    logger.info("Executando RAGAS evaluate() em %d amostras...", len(rows))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
    )

    scores = {
        "faithfulness": float(np.nanmean(result["faithfulness"])),
        "answer_relevancy": float(np.nanmean(result["answer_relevancy"])),
        "context_precision": float(np.nanmean(result["context_precision"])),
        "context_recall": float(np.nanmean(result["context_recall"])),
    }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação RAGAS do pipeline RAG de churn.")
    parser.add_argument("--mode", choices=["direct", "api"], default="direct",
                        help="direct: importa o agente localmente. api: chama POST /analyze.")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="URL base da API (usado com --mode api).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Caminho do JSON de resultados.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limita o número de pares avaliados (útil para testes rápidos).")
    args = parser.parse_args()

    pairs = _load_golden_set()
    rows = build_ragas_dataset(pairs, mode=args.mode, api_url=args.api_url, limit=args.limit)

    if not rows:
        logger.error("Nenhuma linha gerada. Verifique o golden set e as credenciais.")
        sys.exit(1)

    scores = run_ragas_evaluation(rows)

    output_data = {
        "ragas_scores": scores,
        "n_evaluated": len(rows),
        "mode": args.mode,
        "per_pair": [
            {"id": r["pair_id"], "profile_type": r["profile_type"], "answer_preview": r["answer"][:200]}
            for r in rows
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2))

    print("\n=== RAGAS Evaluation Results ===")
    for metric, value in scores.items():
        bar = "█" * int(value * 20)
        print(f"  {metric:<22} {value:.4f}  {bar}")
    print(f"\nResultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
