"""Testes de feature engineering — contratos de schema e qualidade de dados."""
import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from src.features.feature_engineering import (
    build_features,
    convert_types,
    create_derived_features,
    encode_categoricals,
)

# Schema mínimo obrigatório para o output de build_features
OUTPUT_SCHEMA = DataFrameSchema(
    {
        "MonthlyCharges": Column(float, Check.ge(0)),
        "TotalCharges": Column(float, Check.ge(0)),
        "avg_monthly_spend": Column(float, Check.ge(0)),
        "services_count": Column(int, Check.ge(0)),
        "Churn": Column(int, Check.isin([0, 1])),
    },
    coerce=True,
)


# --- convert_types ---


def test_convert_types_total_charges_vira_numerico(telco_raw):
    result = convert_types(telco_raw)
    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])


def test_convert_types_tenure_zero_recebe_monthly_charges(telco_raw):
    """Cliente com tenure=0 (TotalCharges em branco) deve receber MonthlyCharges."""
    result = convert_types(telco_raw)
    tenure_zero = result[result["tenure"] == 0]
    assert not tenure_zero.empty
    assert (tenure_zero["TotalCharges"] == tenure_zero["MonthlyCharges"]).all()


def test_convert_types_nao_altera_linhas_restantes(telco_raw):
    result = convert_types(telco_raw)
    assert result[result["tenure"] > 0]["TotalCharges"].notna().all()


# --- create_derived_features ---


def test_colunas_derivadas_sao_criadas(telco_raw):
    df = convert_types(telco_raw)
    result = create_derived_features(df)
    assert "avg_monthly_spend" in result.columns
    assert "services_count" in result.columns
    assert "tenure_bucket" in result.columns


def test_avg_monthly_spend_tenure_positivo(telco_raw):
    """Para tenure>0, avg_monthly_spend == TotalCharges / tenure."""
    df = convert_types(telco_raw)
    result = create_derived_features(df)
    positivos = result[result["tenure"] > 0]
    esperado = positivos["TotalCharges"] / positivos["tenure"]
    pd.testing.assert_series_equal(
        positivos["avg_monthly_spend"].round(6),
        esperado.round(6),
        check_names=False,
    )


def test_services_count_nao_negativo(telco_raw):
    df = convert_types(telco_raw)
    result = create_derived_features(df)
    assert (result["services_count"] >= 0).all()


def test_tenure_bucket_nao_nulo(telco_raw):
    df = convert_types(telco_raw)
    result = create_derived_features(df)
    assert result["tenure_bucket"].notna().all()


# --- encode_categoricals ---


def test_customer_id_removido(telco_raw):
    df = convert_types(telco_raw)
    df = create_derived_features(df)
    result = encode_categoricals(df)
    assert "customerID" not in result.columns


def test_churn_codificado_como_binario(telco_raw):
    result = build_features(telco_raw)
    assert result["Churn"].isin([0, 1]).all()


# --- build_features (pipeline completo) ---


def test_sem_nulls_apos_build_features(telco_raw):
    result = build_features(telco_raw)
    assert result.isnull().sum().sum() == 0


def test_contagem_de_linhas_preservada(telco_raw):
    result = build_features(telco_raw)
    assert len(result) == len(telco_raw)


def test_sem_colunas_categoricas_remanescentes(telco_raw):
    result = build_features(telco_raw)
    categoricas = result.select_dtypes(include=["object", "category"]).columns.tolist()
    assert categoricas == [], f"Colunas categóricas remanescentes: {categoricas}"


def test_schema_contract(telco_raw):
    """Output de build_features deve respeitar o contrato de schema."""
    result = build_features(telco_raw)
    OUTPUT_SCHEMA.validate(result)
