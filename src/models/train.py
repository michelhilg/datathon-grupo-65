"""Pipeline de treinamento com MLflow tracking padronizado."""
import os
import sys
import logging

import mlflow
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Configurando logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adicionando a raiz do projeto no PYTHONPATH para importar o pacote src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.feature_engineering import build_features

def load_data_and_features(filepath: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Carrega dados brutos e aplica build_features."""
    df = pd.read_csv(filepath)
    df = build_features(df)
    
    # Converte features para float para o scikit-learn, exceto Churn (que é int)
    X = df.drop(columns=["Churn"]).astype(float)
    df = pd.concat([X, df[["Churn"]]], axis=1)
    
    return df

def train_and_log(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    model_class,
    model_params: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> str:
    """Treina modelo, loga tudo no MLflow, retorna run_id.

    Args:
        df: DataFrame com features e target.
        target_col: Nome da coluna target.
        model_name: Nome para registro no MLflow.
        model_class: Classe do modelo (ex: RandomForestClassifier).
        model_params: Hiperparâmetros do modelo.
        test_size: Proporção de teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        run_id do experimento MLflow.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run(run_name=model_name) as run:
        # Log de parâmetros
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])

        # Tags padronizadas (obrigatório)
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("framework", model_class.__module__.split(".")[0])
        mlflow.set_tag("owner", "grupo-65")
        mlflow.set_tag("phase", "datathon-etapa-1")

        # Treino
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Métricas padronizadas
        metrics = {
            "auc": roc_auc_score(y_test, y_pred_proba) if hasattr(model, "predict_proba") else roc_auc_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)

        # Log do modelo
        mlflow.sklearn.log_model(model, "model")

        logger.info(
            "Modelo %s treinado: AUC=%.4f, F1=%.4f",
            model_name,
            metrics["auc"],
            metrics["f1"],
        )

        return run.info.run_id

if __name__ == "__main__":
    # Configura o experimento
    mlflow.set_experiment("Telco_Customer_Churn_Baseline")
    
    logger.info("Carregando e processando os dados...")
    df_processed = load_data_and_features()
    
    logger.info("Iniciando treinamento dos modelos baseline...")
    
    # --- Modelo 1: Regressão Logística ---
    lr_params = {
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 1000
    }
    run_id_lr = train_and_log(
        df=df_processed,
        target_col="Churn",
        model_name="Logistic_Regression",
        model_class=LogisticRegression,
        model_params=lr_params
    )
    
    # --- Modelo 2: Random Forest ---
    rf_params = {
        "class_weight": "balanced",
        "random_state": 42,
        "n_estimators": 100
    }
    run_id_rf = train_and_log(
        df=df_processed,
        target_col="Churn",
        model_name="Random_Forest",
        model_class=RandomForestClassifier,
        model_params=rf_params
    )
    
    logger.info("Todos os treinamentos concluídos com sucesso!")
