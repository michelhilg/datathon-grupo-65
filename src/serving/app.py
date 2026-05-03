"""FastAPI — endpoint de análise de churn via agente ReAct."""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from langchain_core.runnables.config import RunnableConfig
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv()

# Garante que loggers da aplicação (src.*) apareçam no terminal do uvicorn.
# O uvicorn configura apenas seus próprios loggers; o root fica sem handler por padrão.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
)

from src.agent.rag_pipeline import build_index
from src.agent.react_agent import analyze_customer, create_churn_agent
from src.agent.tools import build_tools, churn_predictor
from src.monitoring import (
    CHURN_PROBABILITY_HISTOGRAM,
    DRIFT_PSI_GAUGE,
    DRIFT_RETRAIN_GAUGE,
    SECURITY_BLOCK_COUNTER,
    TOOL_CALL_COUNTER,
    ContextAccumulatorHandler,
    DriftDetector,
    get_langfuse_handler,
    setup_prometheus_middleware,
)
from src.security import InputGuardrail, OutputGuardrail
from src.serving.health import ComponentHealth, overall_status

logger = logging.getLogger(__name__)

_ALLOWED_TOPICS = [
    "churn", "retenção", "retention", "cliente", "customer",
    "contrato", "contract", "serviço", "service", "telecom",
    "internet", "pagamento", "payment", "cancelamento", "cancel",
    "plano", "plan", "fidelidade", "desconto", "upgrade",
]

_app_state: dict[str, Any] = {}


def _load_params() -> dict:
    params_path = Path("params.yaml")
    if params_path.exists():
        with open(params_path) as f:
            return yaml.safe_load(f)
    return {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Inicializa cada subsistema de forma isolada — falha em um não derruba os outros."""
    health: dict[str, ComponentHealth] = {}
    params = _load_params()
    drift_cfg = params.get("drift", {})

    # --- RAG index ---
    rag_health = ComponentHealth("rag")
    health["rag"] = rag_health
    collection = None
    try:
        collection = build_index()
        rag_health.set_ready()
        logger.info("RAG index carregado com sucesso")
    except Exception as exc:
        rag_health.set_failed(str(exc))
        logger.error("RAG index falhou na inicialização: %s", exc)

    # --- Agent (depende do RAG) ---
    agent_health = ComponentHealth("agent")
    health["agent"] = agent_health
    if collection is not None:
        try:
            tools = build_tools(collection)
            agent = create_churn_agent(
                tools=tools,
                model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")),
            )
            _app_state["agent"] = agent
            _app_state["tools"] = tools
            agent_health.set_ready()
            logger.info("Agente inicializado com %d tools", len(tools))
        except Exception as exc:
            agent_health.set_failed(str(exc))
            logger.error("Agente falhou na inicialização: %s", exc)
    else:
        agent_health.set_degraded("RAG index indisponível — /analyze desativado")

    # --- DriftDetector ---
    drift_health = ComponentHealth("drift")
    health["drift"] = drift_health
    try:
        ref_path = Path(drift_cfg.get("reference_path", DriftDetector.DEFAULT_REFERENCE_PATH))
        _app_state["drift_detector"] = DriftDetector(
            window_size=drift_cfg.get("window_size", 500),
            reference_path=ref_path,
            psi_warning=drift_cfg.get("psi_warning"),
            psi_retrain=drift_cfg.get("psi_retrain"),
            min_samples=drift_cfg.get("min_samples"),
        )
        drift_health.set_ready()
    except Exception as exc:
        drift_health.set_failed(str(exc))
        logger.error("DriftDetector falhou na inicialização: %s", exc)

    # --- Guardrails ---
    guardrails_health = ComponentHealth("guardrails")
    health["guardrails"] = guardrails_health
    try:
        _app_state["input_guardrail"] = InputGuardrail(allowed_topics=_ALLOWED_TOPICS)
        _app_state["output_guardrail"] = OutputGuardrail()
        guardrails_health.set_ready()
    except Exception as exc:
        guardrails_health.set_degraded(str(exc))
        logger.warning("Guardrails degradados: %s", exc)
        # Cria guardrails mínimos mesmo em caso de falha parcial
        if "input_guardrail" not in _app_state:
            _app_state["input_guardrail"] = InputGuardrail(allowed_topics=_ALLOWED_TOPICS)
        if "output_guardrail" not in _app_state:
            _app_state["output_guardrail"] = OutputGuardrail()

    _app_state["health"] = health
    logger.info("Startup concluído — status: %s", overall_status(health))
    yield
    _app_state.clear()


app = FastAPI(
    title="Churn Retention Assistant",
    description="Agente ReAct para análise de risco de churn e recomendações de retenção.",
    version="2.0.0",
    lifespan=lifespan,
)

setup_prometheus_middleware(app)


CUSTOMER_EXAMPLE = {
    "customerID": "CUST-0001",
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


class AnalysisRequest(BaseModel):
    customer_features: dict[str, Any] = Field(
        description="Features brutas do cliente no formato Telco dataset.",
        examples=[CUSTOMER_EXAMPLE],
    )
    question: str | None = Field(
        default=None,
        description="Pergunta customizada. Se não informada, usa análise padrão completa.",
    )
    include_contexts: bool = Field(
        default=False,
        description="Inclui chunks RAG recuperados na resposta (útil para avaliação RAGAS).",
    )


class PredictRequest(BaseModel):
    customer_features: dict[str, Any] = Field(
        description="Features brutas do cliente no formato Telco dataset.",
        examples=[CUSTOMER_EXAMPLE],
    )


class PredictResponse(BaseModel):
    churn_probability: float
    prediction: str
    risk_level: str


class AnalysisResponse(BaseModel):
    analysis: str
    customer_id: str | None = None
    contexts: list[str] | None = None


@app.get("/health")
def health():
    """Retorna status granular por componente — degraded não significa downtime total."""
    from src.features.feature_store import get_feature_store

    components = {
        name: h.to_dict()
        for name, h in _app_state.get("health", {}).items()
    }
    feature_store_ok = get_feature_store().ping()
    components["feature_store"] = {
        "status": "ready" if feature_store_ok else "degraded"
    }

    comp_health = _app_state.get("health", {})
    status = overall_status(comp_health) if comp_health else "initializing"

    return {
        "status": status,
        "components": components,
        "capabilities": {
            "predict": True,
            "analyze": "agent" in _app_state,
            "drift_report": "drift_detector" in _app_state,
        },
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expõe métricas Prometheus para scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Chama o modelo ML diretamente — sem agente, sem LLM."""
    customer_json = json.dumps(request.customer_features, ensure_ascii=False)
    try:
        raw = json.loads(churn_predictor.invoke(customer_json))
        if "error" in raw:
            raise HTTPException(status_code=500, detail=raw["error"])

        prob = raw.get("churn_probability", 0.0)
        CHURN_PROBABILITY_HISTOGRAM.observe(prob)
        TOOL_CALL_COUNTER.labels(tool_name="churn_predictor", status="success").inc()

        if "drift_detector" in _app_state:
            _app_state["drift_detector"].record(request.customer_features)

        return PredictResponse(**raw)
    except HTTPException:
        raise
    except Exception as exc:
        TOOL_CALL_COUNTER.labels(tool_name="churn_predictor", status="error").inc()
        logger.error("Erro no /predict: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    """Analisa risco de churn e retorna recomendações de retenção via agente ReAct.

    Com include_contexts=True retorna também os chunks RAG recuperados,
    necessário para avaliação RAGAS.
    """
    if "agent" not in _app_state:
        raise HTTPException(status_code=503, detail="Agente ainda não inicializado.")

    if request.question:
        is_valid, reason = _app_state["input_guardrail"].validate(request.question)
        if not is_valid:
            block_type = "injection" if "suspeito" in reason else "out_of_scope"
            SECURITY_BLOCK_COUNTER.labels(block_type=block_type).inc()
            raise HTTPException(status_code=400, detail=reason)

    customer_json = json.dumps(request.customer_features, ensure_ascii=False)
    customer_id = str(request.customer_features.get("customerID", ""))

    langfuse_cb = get_langfuse_handler()
    ctx_cb = ContextAccumulatorHandler()
    
    callbacks = [ctx_cb]
    if langfuse_cb:
        callbacks.append(langfuse_cb)
        
    config = RunnableConfig(callbacks=callbacks)

    try:
        result = analyze_customer(
            agent=_app_state["agent"],
            customer_json=customer_json,
            question=request.question,
            config=config,
        )

        sanitized = _app_state["output_guardrail"].sanitize(result)

        TOOL_CALL_COUNTER.labels(tool_name="analyze", status="success").inc()

        if "drift_detector" in _app_state:
            _app_state["drift_detector"].record(request.customer_features)

        return AnalysisResponse(
            analysis=sanitized,
            customer_id=customer_id or None,
            contexts=ctx_cb.captured_contexts if request.include_contexts else None,
        )
    except Exception as exc:
        TOOL_CALL_COUNTER.labels(tool_name="analyze", status="error").inc()
        logger.error("Erro na análise: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/drift-report")
def drift_report():
    """Executa detecção de drift Evidently entre dados de referência e predições recentes."""
    if "drift_detector" not in _app_state:
        raise HTTPException(status_code=503, detail="Drift detector não inicializado.")
    try:
        report = _app_state["drift_detector"].run_report()

        if report.get("status") == "ok":
            for feature, metrics in report.get("features", {}).items():
                DRIFT_PSI_GAUGE.labels(feature=feature).set(metrics["psi"])
            DRIFT_RETRAIN_GAUGE.set(1 if report.get("retrain_recommended") else 0)

        return report
    except Exception as exc:
        logger.error("Erro no drift report: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
