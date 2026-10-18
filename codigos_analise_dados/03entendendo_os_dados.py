import pandas as pd 
from pathlib import Path

# ==============================
# CAMINHOS
# ==============================

pasta = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset")

arquivo_entrada = pasta / "atlanta_flights_2015_modelo_19col.csv"
df = pd.read_csv(arquivo_entrada)
pd.set_option("display.max_columns", None)

resumo = pd.DataFrame({
    "tipo": df.dtypes,
    "Faltantes": df.isnull().sum(),
    "percentual_faltantes": df.isnull().mean() * 100,
    "valores_unicos": df.nunique()
})

#print(resumo)
print("========= verificacao de colunas vazias =====================")
print(df.isnull().sum())
print("=============== ve a porcentagem de colunas vazias===============")
print(df.isnull().mean() * 100)
print("=========== minimo tempo de atraso ===================")
print(df[df["AIR_SYSTEM_DELAY"].notna()]["ARRIVAL_DELAY"].min())
print("============= maximo tempo de atraso=================")
print(df[df["AIR_SYSTEM_DELAY"].notna()]["ARRIVAL_DELAY"].max())
print("==============analise estatistica================")
print(df[df["AIR_SYSTEM_DELAY"].notna()]["ARRIVAL_DELAY"].describe())
print("==============================")
print(df.duplicated().sum())