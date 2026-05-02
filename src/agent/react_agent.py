"""Agente ReAct para análise de churn e recomendações de retenção."""
import logging

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Você é um assistente especializado em retenção de clientes de telecomunicações. "
    "Seu objetivo é analisar o risco de churn de clientes e recomendar estratégias de retenção personalizadas. "
    "Sempre use as ferramentas disponíveis para: "
    "(1) calcular a probabilidade de churn com churn_predictor, "
    "(2) identificar os principais fatores de risco com feature_importance, "
    "(3) buscar estratégias de retenção relevantes com retention_knowledge. "
    "Responda em português, de forma estruturada e orientada a ações concretas."
)


def create_churn_agent(
    tools: list,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
):
    """Cria o agente ReAct de análise de churn.

    Args:
        tools: Lista de tools (mínimo 3 obrigatório pelo guia).
        model_name: Modelo OpenAI a utilizar.
        temperature: Temperatura de geração.

    Returns:
        CompiledStateGraph configurado.
    """
    if len(tools) < 3:
        logger.warning("Datathon exige >= 3 tools. Fornecidas: %d", len(tools))

    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def analyze_customer(
    agent,
    customer_json: str,
    question: str | None = None,
) -> str:
    """Executa análise completa de churn para um cliente.

    Args:
        agent: CompiledStateGraph criado por create_churn_agent.
        customer_json: JSON string com features brutas do cliente.
        question: Pergunta customizada (usa padrão se None).

    Returns:
        Análise completa com risco, fatores e estratégias de retenção.
    """
    if not question or not question.strip() or question == "string":
        final_prompt = (
            f"Analise o risco de churn do seguinte cliente e forneça: "
            f"(1) probabilidade de churn, (2) principais fatores de risco, "
            f"(3) estratégias de retenção personalizadas. "
            f"Dados do cliente: {customer_json}"
        )
    else:
        final_prompt = (
            f"O usuário fez a seguinte pergunta sobre retenção de clientes: '{question}'.\n\n"
            f"Por favor, use as ferramentas disponíveis para responder, analisando também os seguintes dados do cliente: {customer_json}"
        )

    result = agent.invoke({"messages": [HumanMessage(content=final_prompt)]})
    return result["messages"][-1].content
