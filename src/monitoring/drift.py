"""Detecção de drift com Evidently e PSI — compara distribuição de referência vs predições recentes."""
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DRIFT_REPORT_PATH = Path("evaluation/drift_report.html")


class DriftDetector:
    """Detecta drift entre a distribuição de treino e as predições recentes.

    Uso:
        detector = DriftDetector(window_size=500)
        detector.record(customer_features_dict)   # chamado a cada predição
        report = detector.run_report()             # gera relatório Evidently + PSI
    """

    DEFAULT_REFERENCE_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
    PSI_WARNING = 0.1
    PSI_RETRAIN = 0.2
    MIN_SAMPLES = 30

    def __init__(
        self,
        window_size: int = 500,
        reference_path: Path | None = None,
        psi_warning: float | None = None,
        psi_retrain: float | None = None,
        min_samples: int | None = None,
    ) -> None:
        self._window: deque[dict] = deque(maxlen=window_size)
        self._reference_path = reference_path or self.DEFAULT_REFERENCE_PATH
        if psi_warning is not None:
            self.PSI_WARNING = psi_warning
        if psi_retrain is not None:
            self.PSI_RETRAIN = psi_retrain
        if min_samples is not None:
            self.MIN_SAMPLES = min_samples
        self._reference_df: pd.DataFrame | None = None

    def record(self, customer_features: dict) -> None:
        """Acumula features de uma predição para análise de drift."""
        record = {}
        for feat in self.FEATURES:
            val = customer_features.get(feat, None)
            if val is not None and str(val).strip() != "":
                try:
                    record[feat] = float(val)
                except (ValueError, TypeError):
                    record[feat] = 0.0
            else:
                record[feat] = 0.0
        self._window.append(record)

    def _load_reference(self) -> pd.DataFrame:
        if self._reference_df is not None:
            return self._reference_df

        if not self._reference_path.exists():
            raise FileNotFoundError(f"Dados de referência não encontrados: {self._reference_path}")

        df = pd.read_csv(self._reference_path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", "0"), errors="coerce").fillna(0.0)
        self._reference_df = df[self.FEATURES].copy()
        return self._reference_df

    @staticmethod
    def _compute_psi(ref: pd.Series, curr: pd.Series, bins: int = 10) -> float:
        """Calcula Population Stability Index (PSI) entre duas distribuições.

        PSI < 0.1  → estável
        PSI 0.1–0.2 → warning (monitorar)
        PSI > 0.2  → drift significativo (considerar retreino)
        """
        ref_clean = ref.dropna()
        curr_clean = curr.dropna()
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return 0.0

        edges = np.histogram_bin_edges(ref_clean, bins=bins)
        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts = np.histogram(ref_clean, bins=edges)[0]
        curr_counts = np.histogram(curr_clean, bins=edges)[0]

        # Suavização para evitar divisão por zero
        ref_pct = (ref_counts + 1e-6) / (len(ref_clean) + 1e-6 * bins)
        curr_pct = (curr_counts + 1e-6) / (len(curr_clean) + 1e-6 * bins)

        psi = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
        return round(psi, 6)

    def run_report(self) -> dict:
        """Executa detecção de drift e retorna relatório com métricas PSI e HTML Evidently.

        Returns:
            Dicionário com status, n_current_samples, métricas por feature e flag retrain_recommended.
        """
        n_samples = len(self._window)
        if n_samples < self.MIN_SAMPLES:
            return {
                "status": "insufficient_data",
                "n_current_samples": n_samples,
                "min_required": self.MIN_SAMPLES,
                "message": f"Necessário no mínimo {self.MIN_SAMPLES} amostras. Atual: {n_samples}.",
            }

        try:
            ref_df = self._load_reference()
        except FileNotFoundError as exc:
            return {"status": "error", "message": str(exc)}

        current_df = pd.DataFrame(list(self._window))

        feature_metrics: dict[str, dict] = {}
        any_retrain = False

        for feat in self.FEATURES:
            if feat not in current_df.columns or feat not in ref_df.columns:
                continue
            psi = self._compute_psi(ref_df[feat], current_df[feat])
            drifted = psi > self.PSI_WARNING
            if psi > self.PSI_RETRAIN:
                any_retrain = True
            feature_metrics[feat] = {
                "psi": psi,
                "drifted": drifted,
                "severity": "retrain" if psi > self.PSI_RETRAIN else ("warning" if drifted else "stable"),
            }

        # Relatório HTML via Evidently (opcional — degrada graciosamente se não instalado)
        report_path: str | None = None
        evidently_available = False
        try:
            from evidently import Report  # noqa: PLC0415
            from evidently.presets import DataDriftPreset  # noqa: PLC0415

            report = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(current_data=current_df, reference_data=ref_df)
            DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(str(DRIFT_REPORT_PATH))
            report_path = str(DRIFT_REPORT_PATH)
            evidently_available = True
            logger.info("Relatório Evidently salvo em %s", report_path)
        except ImportError:
            logger.warning("evidently não instalado — relatório HTML indisponível. PSI calculado manualmente.")
        except Exception as exc:
            logger.warning("Erro ao gerar relatório Evidently: %s", exc)

        return {
            "status": "ok",
            "n_current_samples": n_samples,
            "features": feature_metrics,
            "retrain_recommended": any_retrain,
            "report_path": report_path,
            "evidently_report_generated": evidently_available,
        }
