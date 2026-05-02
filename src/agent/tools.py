"""Tools do agente ReAct — ChurnPredictor, RetentionKnowledge, FeatureImportance."""
import json
import logging
import urllib.parse
from pathlib import Path

import mlflow
import pandas as pd
from langchain_core.tools import tool

from src.features.feature_engineering import build_features

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_best_model():
    """Carrega o modelo RF de maior AUC registrado no MLflow."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Telco_Customer_Churn_Baseline")
    if experiment is None:
        raise RuntimeError("Experimento MLflow não encontrado. Execute src/models/train.py primeiro.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_type = 'classification'",
        order_by=["metrics.auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Nenhum run encontrado no experimento MLflow.")

    run = runs[0]
    artifact_uri = run.info.artifact_uri

    if artifact_uri.startswith("file://"):
        local_path = urllib.parse.unquote(artifact_uri[7:])
    else:
        local_path = artifact_uri

    # Resolve caminho absoluto a partir de __file__ — funciona no host e no Docker.
    # Verifica existência do MLmodel para lidar com duas estruturas possíveis:
    #   - artifacts/MLmodel        (runs antigas, artifact_path vazio)
    #   - artifacts/model/MLmodel  (padrão MLflow com log_model(model, "model"))
    if "/mlruns/" in local_path:
        relative = "mlruns" + local_path.split("/mlruns")[1]
        base = _PROJECT_ROOT / relative
        if (base / "MLmodel").exists():
            model_path = str(base)
        elif (base / "model" / "MLmodel").exists():
            model_path = str(base / "model")
        else:
            model_path = None
    else:
        model_path = None

    uri_to_load = model_path if model_path else f"runs:/{run.info.run_id}/model"

    model = mlflow.sklearn.load_model(uri_to_load)
    logger.info("Modelo carregado: run_id=%s, AUC=%.4f (from %s)", run.info.run_id, run.data.metrics.get("auc", 0), uri_to_load)
    return model


_model_cache: dict = {}


def _get_model():
    if "model" not in _model_cache:
        _model_cache["model"] = _load_best_model()
    return _model_cache["model"]


@tool
def churn_predictor(customer_json: str) -> str:
    """Prevê probabilidade de churn dado um JSON com features brutas do cliente Telco.

    Args:
        customer_json: JSON string com colunas brutas (tenure, MonthlyCharges,
            Contract, InternetService, PaymentMethod, etc.).

    Returns:
        JSON com churn_probability, prediction e risk_level.
    """
    try:
        raw = json.loads(customer_json)
        df_raw = pd.DataFrame([raw])
        df_features = build_features(df_raw)

        model = _get_model()
        feature_cols = [c for c in df_features.columns if c != "Churn"]
        X = df_features[feature_cols].astype(float)

        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)

        prob = float(model.predict_proba(X)[0][1])
        prediction = "Churn" if prob >= 0.5 else "No Churn"
        risk = "Alto" if prob >= 0.7 else ("Médio" if prob >= 0.4 else "Baixo")

        return json.dumps({
            "churn_probability": round(prob, 4),
            "prediction": prediction,
            "risk_level": risk,
        }, ensure_ascii=False)

    except Exception as exc:
        logger.error("Erro no churn_predictor: %s", exc)
        return json.dumps({"error": str(exc)})


@tool
def feature_importance(customer_json: str) -> str:
    """Retorna os 5 principais fatores de risco de churn para um cliente específico.

    Args:
        customer_json: JSON string com colunas brutas do Telco dataset.

    Returns:
        JSON com top_5_risk_factors (feature, importance, value).
    """
    try:
        raw = json.loads(customer_json)
        df_raw = pd.DataFrame([raw])
        df_features = build_features(df_raw)

        model = _get_model()
        feature_cols = [c for c in df_features.columns if c != "Churn"]
        X = df_features[feature_cols].astype(float)

        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)
            feature_cols = list(model.feature_names_in_)

        if not hasattr(model, "feature_importances_"):
            return json.dumps({"error": "Modelo não suporta feature_importances_"})

        factors = sorted(
            [
                {"feature": col, "importance": round(float(imp), 4), "value": round(float(val), 4)}
                for col, imp, val in zip(feature_cols, model.feature_importances_, X.iloc[0].values)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )[:5]

        return json.dumps({"top_5_risk_factors": factors}, ensure_ascii=False)

    except Exception as exc:
        logger.error("Erro no feature_importance: %s", exc)
        return json.dumps({"error": str(exc)})


def make_retention_knowledge_tool(collection):
    """Factory que injeta a coleção ChromaDB na closure da tool."""
    from src.agent.rag_pipeline import retrieve

    @tool
    def retention_knowledge(query: str) -> str:
        """Busca estratégias de retenção, padrões de churn e recomendações de produto.

        Args:
            query: Pergunta em linguagem natural sobre retenção de clientes,
                padrões de churn ou recomendações de serviço.

        Returns:
            Trechos relevantes da knowledge base com estratégias e benchmarks.
        """
        try:
            return retrieve(query=query, collection=collection)
        except Exception as exc:
            logger.error("Erro no retention_knowledge: %s", exc)
            return f"Erro ao consultar knowledge base: {exc}"

    return retention_knowledge


def build_tools(collection) -> list:
    """Constrói a lista de tools para o agente ReAct.

    Args:
        collection: Coleção ChromaDB já indexada.

    Returns:
        Lista com as 3 tools obrigatórias.
    """
    return [
        churn_predictor,
        retention_knowledge := make_retention_knowledge_tool(collection),
        feature_importance,
    ]
