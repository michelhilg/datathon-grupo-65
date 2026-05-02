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
import shutil
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_EXPORT_DIR = PROJECT_ROOT / "model"


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
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("Telco_Customer_Churn_Baseline")
    if experiment is None:
        raise RuntimeError(
            "Experimento 'Telco_Customer_Churn_Baseline' não encontrado. "
            "Execute: uv run python -m src.models.train"
        )

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

    if MODEL_EXPORT_DIR.exists():
        shutil.rmtree(MODEL_EXPORT_DIR)
    shutil.copytree(model_src, MODEL_EXPORT_DIR)

    print(f"run_id : {run.info.run_id}")
    print(f"AUC    : {auc:.4f}")
    print(f"origem : {model_src}")
    print(f"destino: {MODEL_EXPORT_DIR}")
    print("Modelo exportado. Agora execute: docker compose build && docker compose up")


if __name__ == "__main__":
    export_best_model()
