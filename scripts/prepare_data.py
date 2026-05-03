"""Stage: carrega dados brutos, aplica feature engineering e salva como parquet."""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.features.feature_engineering import build_features


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    raw_path = params["data"]["raw_path"]
    processed_path = params["data"]["processed_path"]

    df = pd.read_csv(raw_path)
    df = build_features(df)

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    print(f"Features salvas: {processed_path} ({len(df)} linhas, {df.shape[1]} colunas)")


if __name__ == "__main__":
    main()
