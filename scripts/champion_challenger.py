"""Champion-Challenger evaluation para retraining automatizado.

Fluxo:
  1. Carrega métricas do champion de evaluation/model_metrics.json
  2. Treina LR + RF como challenger no experimento de retraining
  3. Compara o melhor challenger contra o champion no mesmo holdout
  4. Decisão: promote se delta_auc >= params.champion_challenger.min_delta_auc
  5. Se promote (ou bootstrap): exporta challenger para model/ e
     atualiza evaluation/model_metrics.json

Saída:
  evaluation/champion_challenger_report.json  — relatório detalhado
  model/                                       — challenger exportado (promote|bootstrap)
  evaluation/model_metrics.json               — atualizado (promote|bootstrap)

GitHub Actions: escreve decision/challenger_run_id/delta_auc em GITHUB_OUTPUT.
Código de saída: sempre 0 (decisão lida do relatório/output, não do exit code).
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import mlflow
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import load_data_and_features, train_and_log  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_params(base_path: Path = PROJECT_ROOT) -> dict:
    with open(base_path / "params.yaml") as f:
        return yaml.safe_load(f)


def _load_champion_metrics(base_path: Path = PROJECT_ROOT) -> dict | None:
    """Lê métricas do champion de evaluation/model_metrics.json.

    Retorna None se o arquivo não existir ou se validation_passed for False,
    sinalizando que o pipeline deve operar em modo bootstrap.
    """
    path = base_path / "evaluation" / "model_metrics.json"
    if not path.exists():
        logger.warning("evaluation/model_metrics.json não encontrado — modo bootstrap.")
        return None
    data = json.loads(path.read_text())
    if not data.get("validation_passed"):
        logger.warning("Champion sem validation_passed=True — modo bootstrap.")
        return None
    return data


def _decide(champion_auc: float, challenger_auc: float, threshold: float) -> str:
    """Retorna 'promote' se delta_auc >= threshold, 'skip' caso contrário."""
    return "promote" if (challenger_auc - champion_auc) >= threshold else "skip"


def _train_challenger(params: dict, experiment_name: str) -> dict:
    """Treina LR e RF no experimento de retraining e retorna métricas do melhor."""
    data_cfg = params.get("data", {})
    model_cfg = params.get("model", {})
    training_cfg = params.get("training", {})

    raw_path = data_cfg.get("raw_path", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    test_size = model_cfg.get("test_size", 0.2)
    random_state = model_cfg.get("random_state", 42)
    lr_cfg = training_cfg.get("logistic_regression", {})
    rf_cfg = training_cfg.get("random_forest", {})

    mlflow.set_experiment(experiment_name)
    df = load_data_and_features(filepath=raw_path)

    lr_params = {
        "class_weight": lr_cfg.get("class_weight", "balanced"),
        "random_state": lr_cfg.get("random_state", random_state),
        "max_iter": lr_cfg.get("max_iter", 1000),
    }
    run_id_lr = train_and_log(
        df=df,
        target_col="Churn",
        model_name="Challenger_LR",
        model_class=LogisticRegression,
        model_params=lr_params,
        test_size=test_size,
        random_state=random_state,
    )

    rf_params = {
        "class_weight": rf_cfg.get("class_weight", "balanced"),
        "random_state": rf_cfg.get("random_state", random_state),
        "n_estimators": rf_cfg.get("n_estimators", 100),
    }
    run_id_rf = train_and_log(
        df=df,
        target_col="Churn",
        model_name="Challenger_RF",
        model_class=RandomForestClassifier,
        model_params=rf_params,
        test_size=test_size,
        random_state=random_state,
    )

    client = mlflow.tracking.MlflowClient()
    candidates = []
    for run_id, name in [(run_id_lr, "Challenger_LR"), (run_id_rf, "Challenger_RF")]:
        run = client.get_run(run_id)
        candidates.append(
            {
                "name": name,
                "run_id": run_id,
                "experiment_id": run.info.experiment_id,
                "auc": round(run.data.metrics.get("auc", 0.0), 4),
                "f1": round(run.data.metrics.get("f1", 0.0), 4),
                "precision": round(run.data.metrics.get("precision", 0.0), 4),
                "recall": round(run.data.metrics.get("recall", 0.0), 4),
            }
        )

    best = max(candidates, key=lambda r: r["auc"])
    logger.info("Melhor challenger: %s  AUC=%.4f", best["name"], best["auc"])
    return best


def _export_challenger(
    challenger: dict, params: dict, base_path: Path = PROJECT_ROOT
) -> None:
    """Copia artefatos do challenger para model/ — replica lógica do export_model.py."""
    export_dir = base_path / params.get("model", {}).get("export_dir", "model")
    experiment_id = challenger["experiment_id"]
    run_id = challenger["run_id"]

    models_root = base_path / "mlruns" / experiment_id / "models"
    if not models_root.exists():
        raise RuntimeError(f"mlruns não encontrado: {models_root}")

    model_src = None
    for candidate in models_root.iterdir():
        mlmodel_file = candidate / "artifacts" / "MLmodel"
        if not mlmodel_file.exists():
            continue
        for line in mlmodel_file.read_text().splitlines():
            if line.strip().startswith("run_id:") and line.split(":", 1)[1].strip() == run_id:
                model_src = candidate / "artifacts"
                break
        if model_src:
            break

    if model_src is None:
        raise RuntimeError(f"Modelo run_id={run_id} não encontrado em {models_root}")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(model_src, export_dir)
    logger.info("Challenger exportado: %s → %s", model_src, export_dir)


def _write_updated_metrics(
    challenger: dict, params: dict, base_path: Path = PROJECT_ROOT
) -> None:
    """Atualiza evaluation/model_metrics.json com os dados do challenger promovido."""
    eval_dir = base_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cc_cfg = params.get("champion_challenger", {})
    metrics = {
        "experiment_name": cc_cfg.get("experiment_name", "Telco_Customer_Churn_Retraining"),
        "run_id": challenger["run_id"],
        "model_name": challenger["name"],
        "auc": challenger["auc"],
        "f1": challenger["f1"],
        "precision": challenger["precision"],
        "recall": challenger["recall"],
        "min_auc_threshold": params["model"]["min_auc_threshold"],
        "validation_passed": challenger["auc"] >= params["model"]["min_auc_threshold"],
        "promoted_by": "champion_challenger",
    }
    (eval_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("evaluation/model_metrics.json atualizado com challenger promovido.")


def _write_github_output(**kwargs: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a") as f:
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    params = _load_params()
    cc_cfg = params.get("champion_challenger", {})
    delta_threshold = cc_cfg.get("min_delta_auc", 0.005)
    experiment_name = cc_cfg.get("experiment_name", "Telco_Customer_Churn_Retraining")

    champion = _load_champion_metrics()
    challenger = _train_challenger(params, experiment_name)

    if champion is None:
        delta_auc = 0.0
        min_auc = params["model"]["min_auc_threshold"]
        if challenger["auc"] < min_auc:
            decision = "skip"
            logger.warning(
                "Bootstrap ABORTADO: challenger AUC=%.4f abaixo do threshold mínimo %.4f"
                " — nenhum modelo promovido.",
                challenger["auc"],
                min_auc,
            )
        else:
            decision = "bootstrap"
            logger.info("Bootstrap: nenhum champion existente — challenger vira champion inicial.")
            _export_challenger(challenger, params)
            _write_updated_metrics(challenger, params)
    else:
        champion_auc = champion.get("auc", 0.0)
        logger.info(
            "Champion: %s  AUC=%.4f  run_id=%s",
            champion.get("model_name"),
            champion_auc,
            champion.get("run_id"),
        )
        delta_auc = round(challenger["auc"] - champion_auc, 6)
        decision = _decide(champion_auc, challenger["auc"], delta_threshold)

        if decision == "promote":
            logger.info(
                "PROMOTE: challenger supera champion em +%.4f AUC (threshold: %.4f)",
                delta_auc,
                delta_threshold,
            )
            _export_challenger(challenger, params)
            _write_updated_metrics(challenger, params)
        else:
            logger.info(
                "SKIP: delta AUC %.4f abaixo do threshold %.4f — champion mantido.",
                delta_auc,
                delta_threshold,
            )

    report = {
        "decision": decision,
        "delta_auc": delta_auc,
        "threshold": delta_threshold,
        "champion": {
            "run_id": champion.get("run_id") if champion else None,
            "model_name": champion.get("model_name") if champion else None,
            "auc": champion.get("auc") if champion else None,
        },
        "challenger": {
            "run_id": challenger["run_id"],
            "model_name": challenger["name"],
            "auc": challenger["auc"],
            "f1": challenger["f1"],
        },
    }

    eval_dir = PROJECT_ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "champion_challenger_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Relatório gravado em evaluation/champion_challenger_report.json")

    _write_github_output(
        decision=decision,
        challenger_run_id=challenger["run_id"],
        delta_auc=str(delta_auc),
    )


if __name__ == "__main__":
    main()
