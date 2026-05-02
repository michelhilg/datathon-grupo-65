"""LLM-as-judge para avaliação qualitativa das respostas do agente de retenção.

Critérios de avaliação (escala 0–10):
  1. fidelidade_aos_dados      — A resposta cita corretamente probabilidade de churn e fatores de risco?
  2. relevancia_estrategica    — As estratégias são pertinentes ao perfil específico do cliente?
  3. valor_acao_retencao       — (critério de negócio) As ações são concretas, executáveis e com ROI positivo?
  4. clareza_estrutura         — A resposta é bem estruturada, em português e compreensível para analistas?

Uso:
    uv run python evaluation/llm_judge.py [--limit N] [--output PATH]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/golden_set/golden_set.json")
DEFAULT_OUTPUT = Path("evaluation/llm_judge_results.json")

JUDGE_PROMPT_TEMPLATE = """Você é um especialista em retenção de clientes de telecomunicações e avaliador de sistemas de IA.

Avalie a resposta do agente abaixo usando a escala 0-10 para cada critério.

## Perfil do Cliente
{customer_summary}

## Pergunta Feita ao Agente
{question}

## Resposta do Agente
{answer}

## Resposta de Referência Esperada
{ground_truth}

## Critérios de Avaliação

1. **fidelidade_aos_dados** (0-10): A resposta cita corretamente a probabilidade de churn e os fatores de risco identificados pelas ferramentas? Penalize se inventar dados ou ignorar o que as ferramentas retornaram.

2. **relevancia_estrategica** (0-10): As estratégias de retenção recomendadas são pertinentes ao perfil específico deste cliente (nível de risco, tipo de contrato, serviços contratados, tenure)? Penalize recomendações genéricas que não consideram o perfil.

3. **valor_acao_retencao** (0-10): As ações recomendadas são concretas e executáveis por um agente de atendimento, com ROI positivo esperado dado o perfil do cliente? Exemplos de alta pontuação: desconto com percentual específico, oferta de serviço com prazo, script de abordagem. Exemplos de baixa pontuação: "entre em contato com o cliente", "ofereça algo".

4. **clareza_estrutura** (0-10): A resposta é bem estruturada, redigida em português, com seções claras e compreensível por um analista de negócios sem conhecimento técnico de ML?

## Formato de Resposta

Responda APENAS com um JSON válido, sem markdown, sem explicações fora do JSON:

{{"fidelidade_aos_dados": <0-10>, "relevancia_estrategica": <0-10>, "valor_acao_retencao": <0-10>, "clareza_estrutura": <0-10>, "justificativa": "<1-2 frases explicando os pontos mais críticos da avaliação>"}}"""


def _customer_summary(features: dict) -> str:
    """Gera resumo legível das features do cliente para o prompt do juiz."""
    parts = [
        f"Tenure: {features.get('tenure', '?')} meses",
        f"Cobrança mensal: R${features.get('MonthlyCharges', '?')}",
        f"Contrato: {features.get('Contract', '?')}",
        f"Internet: {features.get('InternetService', '?')}",
        f"Pagamento: {features.get('PaymentMethod', '?')}",
        f"Idoso: {'Sim' if features.get('SeniorCitizen') == 1 else 'Não'}",
    ]
    return " | ".join(parts)


def parse_judge_response(raw: str) -> dict:
    """Extrai scores do JSON retornado pelo LLM juiz. Retorna zeros em caso de falha."""
    fallback = {
        "fidelidade_aos_dados": 0,
        "relevancia_estrategica": 0,
        "valor_acao_retencao": 0,
        "clareza_estrutura": 0,
        "justificativa": "[parse error]",
        "_parse_error": True,
    }
    try:
        # Remove possíveis blocos markdown
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(clean)
        for key in ("fidelidade_aos_dados", "relevancia_estrategica", "valor_acao_retencao", "clareza_estrutura"):
            if key not in data:
                return {**fallback, "_parse_error": True, "_raw": raw[:500]}
        return data
    except (json.JSONDecodeError, KeyError):
        return {**fallback, "_parse_error": True, "_raw": raw[:500]}


def _call_judge(prompt: str, model: str, client) -> dict:
    """Chama o LLM juiz com retry exponencial (3 tentativas)."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            return parse_judge_response(raw)
        except Exception as exc:
            if attempt == 2:
                logger.error("Falha após 3 tentativas: %s", exc)
                return parse_judge_response("")
            wait = 2 ** attempt
            logger.warning("Tentativa %d falhou (%s). Aguardando %ds...", attempt + 1, exc, wait)
            time.sleep(wait)
    return parse_judge_response("")


def _get_agent_answer(pair: dict) -> str:
    """Executa o agente localmente e retorna a resposta."""
    import json as _json  # noqa: PLC0415

    from langchain_core.runnables.config import RunnableConfig  # noqa: PLC0415

    from src.agent.rag_pipeline import build_index  # noqa: PLC0415
    from src.agent.react_agent import analyze_customer  # noqa: PLC0415
    from src.agent.tools import build_tools  # noqa: PLC0415

    if not hasattr(_get_agent_answer, "_agent"):
        logger.info("Inicializando agente para LLM judge...")
        collection = build_index()
        tools = build_tools(collection)
        from src.agent.react_agent import create_churn_agent  # noqa: PLC0415
        _get_agent_answer._agent = create_churn_agent(  # type: ignore[attr-defined]
            tools=tools,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.0,
        )

    customer_json = _json.dumps(pair["customer_features"], ensure_ascii=False)
    return analyze_customer(
        agent=_get_agent_answer._agent,  # type: ignore[attr-defined]
        customer_json=customer_json,
        question=pair.get("question"),
        config=RunnableConfig(callbacks=[]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge para avaliação do agente de retenção.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limita o número de pares avaliados.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Caminho do JSON de resultados.")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Modelo OpenAI usado como juiz.")
    args = parser.parse_args()

    try:
        import openai  # noqa: PLC0415
    except ImportError:
        logger.error("openai não instalado. Execute: uv add openai")
        sys.exit(1)

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(GOLDEN_SET_PATH) as f:
        golden_set = json.load(f)["pairs"]

    if args.limit:
        golden_set = golden_set[:args.limit]

    logger.info("Iniciando LLM-judge em %d pares com modelo %s", len(golden_set), args.judge_model)

    results = []
    criteria_sums: dict[str, float] = {
        "fidelidade_aos_dados": 0.0,
        "relevancia_estrategica": 0.0,
        "valor_acao_retencao": 0.0,
        "clareza_estrutura": 0.0,
    }
    valid_count = 0

    for i, pair in enumerate(golden_set):
        logger.info("[%d/%d] Avaliando par %s...", i + 1, len(golden_set), pair["id"])
        try:
            answer = _get_agent_answer(pair)
        except Exception as exc:
            logger.error("Erro ao obter resposta do agente para %s: %s", pair["id"], exc)
            continue

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_summary=_customer_summary(pair["customer_features"]),
            question=pair["question"],
            answer=answer,
            ground_truth=pair["ground_truth"],
        )

        scores = _call_judge(prompt, model=args.judge_model, client=client)

        result = {
            "id": pair["id"],
            "profile_type": pair["profile_type"],
            "scores": scores,
            "answer_preview": answer[:300],
        }
        results.append(result)

        if not scores.get("_parse_error"):
            valid_count += 1
            for key in criteria_sums:
                criteria_sums[key] += float(scores.get(key, 0))

    averages = {k: round(v / valid_count, 4) for k, v in criteria_sums.items()} if valid_count > 0 else {}

    output_data = {
        "judge_model": args.judge_model,
        "n_evaluated": len(results),
        "n_valid": valid_count,
        "averages": averages,
        "per_pair": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2))

    print("\n=== LLM-as-Judge Results ===")
    criteria_labels = {
        "fidelidade_aos_dados": "Fidelidade aos dados   ",
        "relevancia_estrategica": "Relevância estratégica ",
        "valor_acao_retencao": "Valor ação retenção    ",
        "clareza_estrutura": "Clareza e estrutura    ",
    }
    for key, label in criteria_labels.items():
        val = averages.get(key, 0)
        bar = "█" * int(val * 2)
        print(f"  {label} {val:.2f}/10  {bar}")
    print(f"\nAvaliados: {valid_count}/{len(results)} pares")
    print(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
