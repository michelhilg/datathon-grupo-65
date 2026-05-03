"""Exporta o melhor modelo do MLflow para ./model/ antes do docker build.

Uso:
    python scripts/export_model.py

O diretório ./model/ gerado é copiado para /app/model dentro da imagem Docker
via COPY . . (model/ não está no .dockerignore). O container carrega o modelo
de MODEL_PATH=/app/model sem depender de paths absolutos do host.

Compatível com MLflow 3.x, que salva artefatos em
mlruns/{exp_id}/models/m-{model_id}/artifacts/ em vez de
mlruns/{exp_id}/{run_id}/artifacts/.
"""
import json
import shutil
from pathlib import Path

import mlflow
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_params() -> dict:
    params_path = PROJECT_ROOT / "params.yaml"
    if params_path.exists():
        with open(params_path) as f:
            return yaml.safe_load(f)
    return {}


def _get_export_dir(params: dict) -> Path:
    export_dir = params.get("model", {}).get("export_dir", "model")
    return PROJECT_ROOT / export_dir


def _find_model_dir(run_id: str, experiment_id: str) -> Path:
    """Localiza o diretório do modelo escaneando MLmodel files por run_id.

    MLflow 3.x não salva artefatos em mlruns/{exp}/{run_id}/artifacts/.
    Em vez disso usa mlruns/{exp}/models/m-{model_id}/artifacts/, onde o
    run_id está registrado dentro do próprio MLmodel.
    """
    models_root = PROJECT_ROOT / "mlruns" / experiment_id / "models"
    if not models_root.exists():
        raise RuntimeError(
            f"Diretório de modelos não encontrado: {models_root}. "
            "Verifique se o treinamento foi concluído (src/models/train.py)."
        )

    for candidate in models_root.iterdir():
        mlmodel_file = candidate / "artifacts" / "MLmodel"
        if not mlmodel_file.exists():
            continue
        for line in mlmodel_file.read_text().splitlines():
            if line.strip().startswith("run_id:"):
                found = line.split(":", 1)[1].strip()
                if found == run_id:
                    return candidate / "artifacts"

    raise RuntimeError(
        f"Modelo para run_id={run_id} não encontrado em {models_root}. "
        "Verifique se o treinamento foi concluído (src/models/train.py)."
    )


def export_best_model() -> None:
    params = _load_params()
    export_dir = _get_export_dir(params)
    experiment_name = params.get("model", {}).get("experiment_name", "Telco_Customer_Churn_Baseline")

    # Se evaluation/model_metrics.json existir (produzido pelo stage evaluate),
    # usa o run_id validado em vez de re-consultar o MLflow.
    metrics_path = PROJECT_ROOT / "evaluation" / "model_metrics.json"
    validated_run_id: str | None = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("validation_passed"):
            validated_run_id = metrics.get("run_id")
            print(f"Usando run_id validado pelo stage evaluate: {validated_run_id}")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experimento '{experiment_name}' não encontrado. "
            "Execute: uv run python src/models/train.py"
        )

    if validated_run_id:
        run = client.get_run(validated_run_id)
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_type = 'classification'",
            order_by=["metrics.auc DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("Nenhum run com tag model_type='classification' encontrado.")
        run = runs[0]

    auc = run.data.metrics.get("auc", 0)
    model_src = _find_model_dir(run.info.run_id, run.info.experiment_id)

    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(model_src, export_dir)

    print(f"run_id : {run.info.run_id}")
    print(f"AUC    : {auc:.4f}")
    print(f"origem : {model_src}")
    print(f"destino: {export_dir}")
    print("Modelo exportado. Agora execute: docker compose build && docker compose up")


if __name__ == "__main__":
    export_best_model()
