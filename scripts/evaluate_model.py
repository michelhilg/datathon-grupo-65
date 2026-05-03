"""Stage: valida métricas do melhor modelo antes de exportar.

Lê training_output.json produzido pelo stage train e verifica se o AUC
do melhor modelo atinge o threshold mínimo definido em params.yaml.
Falha com exit code 1 se a validação não passar — bloqueia export_model no DAG.
"""
import json
import sys
from pathlib import Path

import mlflow
import yaml


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    min_auc = params["model"]["min_auc_threshold"]
    experiment_name = params["model"]["experiment_name"]

    training_output = json.loads(Path("training_output.json").read_text())
    run_id = training_output["best_run_id"]

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    auc = round(run.data.metrics.get("auc", 0.0), 4)
    f1 = round(run.data.metrics.get("f1", 0.0), 4)
    precision = round(run.data.metrics.get("precision", 0.0), 4)
    recall = round(run.data.metrics.get("recall", 0.0), 4)

    metrics = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "model_name": training_output.get("best_model", "unknown"),
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "min_auc_threshold": min_auc,
        "validation_passed": auc >= min_auc,
    }

    Path("evaluation").mkdir(parents=True, exist_ok=True)
    Path("evaluation/model_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Modelo avaliado: {metrics['model_name']} (run_id={run_id})")
    print(f"  AUC={auc:.4f}  F1={f1:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")

    if not metrics["validation_passed"]:
        print(f"FALHA: AUC {auc:.4f} está abaixo do threshold mínimo {min_auc}")
        sys.exit(1)

    print(f"Validação OK: AUC {auc:.4f} >= threshold {min_auc}")


if __name__ == "__main__":
    main()
