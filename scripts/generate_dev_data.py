"""Stage: gera dataset sintético para dev/test — isolado do pipeline de produção.

Produz data/dev/synthetic_telco.csv com distribuições correlacionadas:
- Churn mais provável para tenure curto e contrato mensal
- Fibra óptica associada a maior churn rate
- Seed fixo garante reprodutibilidade entre runs
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from tests.fixtures.synthetic_data import generate_synthetic_telco


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    dev_path = Path(params["data"]["dev_path"])
    dev_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_telco(n_rows=500, seed=42)
    df.to_csv(dev_path, index=False)

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Dataset sintético gerado : {dev_path}")
    print(f"  Linhas                 : {len(df)}")
    print(f"  Colunas                : {df.shape[1]}")
    print(f"  Churn rate             : {churn_rate:.1%}")
    print(f"  tenure range           : [{df['tenure'].min()}, {df['tenure'].max()}]")
    print(f"  MonthlyCharges range   : [{df['MonthlyCharges'].min():.2f}, {df['MonthlyCharges'].max():.2f}]")


if __name__ == "__main__":
    main()
