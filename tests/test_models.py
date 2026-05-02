"""Testes do pipeline de treinamento — determinismo, métricas e governança."""
import mlflow
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.feature_engineering import build_features
from src.models.train import train_and_log


@pytest.fixture
def mlflow_tracking(tmp_path):
    """MLflow com tracking URI isolado — não polui mlruns/ local."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/test_mlflow.db")
    mlflow.set_experiment("test_churn_experiment")
    yield
    mlflow.end_run()


@pytest.fixture
def processed_data(telco_raw):
    """DataFrame já processado pelo pipeline de features."""
    return build_features(telco_raw)


# --- train_and_log ---


def test_retorna_run_id_valido(processed_data, mlflow_tracking):
    run_id = train_and_log(
        df=processed_data,
        target_col="Churn",
        model_name="Test_LR",
        model_class=LogisticRegression,
        model_params={"max_iter": 200, "random_state": 42},
    )
    assert isinstance(run_id, str)
    assert len(run_id) == 32  # MLflow run IDs são strings hex de 32 chars


def test_metricas_no_intervalo_valido(processed_data, mlflow_tracking):
    """Todas as métricas logadas devem estar em [0.0, 1.0]."""
    run_id = train_and_log(
        df=processed_data,
        target_col="Churn",
        model_name="Test_Metrics",
        model_class=LogisticRegression,
        model_params={"max_iter": 200, "random_state": 42},
    )
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id).data.metrics

    for nome, valor in metrics.items():
        assert 0.0 <= valor <= 1.0, f"Métrica '{nome}' = {valor} está fora de [0, 1]"


def test_tags_obrigatorias_de_governanca(processed_data, mlflow_tracking):
    """Run deve conter as tags padronizadas exigidas pelo guia."""
    run_id = train_and_log(
        df=processed_data,
        target_col="Churn",
        model_name="Test_Tags",
        model_class=LogisticRegression,
        model_params={"max_iter": 200, "random_state": 42},
    )
    client = mlflow.tracking.MlflowClient()
    tags = client.get_run(run_id).data.tags

    assert "model_type" in tags
    assert "owner" in tags
    assert "phase" in tags
    assert tags["model_type"] == "classification"
    assert tags["owner"] == "grupo-65"


def test_parametros_logados(processed_data, mlflow_tracking):
    """Parâmetros de treino devem estar registrados no run."""
    run_id = train_and_log(
        df=processed_data,
        target_col="Churn",
        model_name="Test_Params",
        model_class=LogisticRegression,
        model_params={"max_iter": 200, "random_state": 42},
    )
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params

    assert "test_size" in params
    assert "random_state" in params
    assert "n_features" in params
    assert "n_samples_train" in params
