import pandas as pd
import numpy as np

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte a coluna 'TotalCharges' para numérico, preenchendo 
    valores faltantes (onde tenure=0) com 'MonthlyCharges'.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Para clientes recém-chegados (tenure=0), TotalCharges vem vazio.
    # Usamos o MonthlyCharges como primeira cobrança (aproximação).
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    
    return df

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria as features derivadas: avg_monthly_spend, services_count, tenure_bucket.
    """
    df = df.copy()
    
    # 1. avg_monthly_spend
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, 
        df["TotalCharges"] / df["tenure"], 
        df["MonthlyCharges"]
    )
    
    # 2. services_count
    # Lista de serviços extras que possuem a flag "Yes"
    yes_no_services = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Conta quantos desses serviços o cliente tem
    df["services_count"] = (df[yes_no_services] == "Yes").sum(axis=1)
    
    # Soma também se tiver serviço de internet ('DSL' ou 'Fiber optic')
    if "InternetService" in df.columns:
        df["services_count"] += (df["InternetService"] != "No").astype(int)
        
    # 3. tenure_bucket
    bins = [-1, 12, 24, 36, 48, 60, np.inf]
    labels = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", ">60m"]
    # Adicionamos como string/category para que o get_dummies capture depois
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)
    
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica encoding binário e one-hot encoding nas variáveis categóricas.
    """
    df = df.copy()
    
    # Remove customerID se existir, pois não é feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    # Mapeamento binário
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    if "Churn" in df.columns:
        binary_cols.append("Churn")
        
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
            
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 1, "Male": 0})
        
    # Pega o restante das colunas categóricas (object ou string)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Aplica One-Hot Encoding (drop_first=True para evitar multicolinearidade)
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa todo o pipeline de engenharia de features.
    """
    df = convert_types(df)
    df = create_derived_features(df)
    df = encode_categoricals(df)
    return df
