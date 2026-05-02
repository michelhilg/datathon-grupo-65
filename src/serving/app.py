"""FastAPI — endpoint de análise de churn via agente ReAct."""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from src.agent.rag_pipeline import build_index
from src.agent.react_agent import analyze_customer, create_churn_agent
from src.agent.tools import build_tools

logger = logging.getLogger(__name__)

# Estado global da aplicação (inicializado no lifespan)
_app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa RAG e agente na subida da aplicação."""
    logger.info("Inicializando knowledge base e agente...")
    collection = build_index()
    tools = build_tools(collection)
    agent = create_churn_agent(
        tools=tools,
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")),
    )
    _app_state["agent"] = agent
    _app_state["tools"] = tools
    logger.info("Agente inicializado com %d tools", len(tools))
    yield
    _app_state.clear()


app = FastAPI(
    title="Churn Retention Assistant",
    description="Agente ReAct para análise de risco de churn e recomendações de retenção.",
    version="1.0.0",
    lifespan=lifespan,
)


class AnalysisRequest(BaseModel):
    customer_features: dict[str, Any] = Field(
        description="Features brutas do cliente no formato Telco dataset.",
        examples=[{
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
        }]
    )
    question: str | None = Field(
        default=None,
        description="Pergunta customizada. Se não informada, usa análise padrão completa.",
    )


class AnalysisResponse(BaseModel):
    analysis: str
    customer_id: str | None = None


@app.get("/health")
def health():
    """Verifica se a aplicação está pronta para receber requisições."""
    ready = "agent" in _app_state
    return {"status": "ok" if ready else "initializing", "agent_ready": ready}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    """Analisa risco de churn e retorna recomendações de retenção.

    Executa o agente ReAct com as 3 tools:
    - ChurnPredictor: probabilidade e nível de risco
    - RetentionKnowledge: estratégias baseadas na knowledge base
    - FeatureImportance: principais fatores de risco do cliente
    """
    if "agent" not in _app_state:
        raise HTTPException(status_code=503, detail="Agente ainda não inicializado.")

    customer_json = json.dumps(request.customer_features, ensure_ascii=False)
    customer_id = str(request.customer_features.get("customerID", ""))

    try:
        result = analyze_customer(
            agent=_app_state["agent"],
            customer_json=customer_json,
            question=request.question,
        )
        return AnalysisResponse(analysis=result, customer_id=customer_id or None)
    except Exception as exc:
        logger.error("Erro na análise: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
