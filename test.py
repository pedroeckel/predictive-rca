import pandas as pd
import sys
sys.path.append('src')

from src.preprocessing.build_features import build_case_features

# Carrega e processa seu CSV
df = pd.read_csv("data/raw/amarelo.csv")
print("🔍 Dataset original:")
print(f"Shape: {df.shape}")
print(f"Colunas: {df.columns.tolist()}")

# Constrói as features
df_cases = build_case_features(df, sla_hours=56)
print(f"\n📊 Dataset após feature engineering: {df_cases.shape}")

# Verifica NaN
print(f"\n❌ VALORES NaN POR COLUNA:")
print(df_cases.isnull().sum())

# Mostra as linhas com NaN
nan_rows = df_cases[df_cases.isnull().any(axis=1)]
print(f"\n📝 Linhas com NaN ({len(nan_rows)}):")
print(nan_rows)