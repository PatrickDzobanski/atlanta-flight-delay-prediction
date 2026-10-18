import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# CAMINHOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
)

arquivo_entrada = (
    pasta
    / "dataset"
    / "atlanta_flights_com_horario_sem_cancelamento_periodo.csv"
)

arquivo_treino = (
    pasta
    / "dataset_filtrado"
    / "arquivo_treino.csv"
)

arquivo_teste = (
    pasta
    / "dataset_filtrado"
    / "arquivo_teste.csv"
)


# ============================================================
# CARREGAR DATASET
# ============================================================

df = pd.read_csv(arquivo_entrada)


# ============================================================
# COLUNAS QUE SERÃO USADAS
# ============================================================

features = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "DESTINATION_AIRPORT",
    "HOUR",
    "DEPARTURE_MINUTES",
    "SCHEDULED_TIME",
    "DISTANCE"
]


# ============================================================
# X E y
# ============================================================

X = df[features].copy()

y = df["DELAYED"].copy()


# ============================================================
# SEPARAR TREINO E TESTE
# ============================================================

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# JUNTAR X E y NOVAMENTE
# ============================================================

df_treino = pd.concat(
    [X_treino, y_treino],
    axis=1
)

df_teste = pd.concat(
    [X_teste, y_teste],
    axis=1
)


# ============================================================
# SALVAR CSVs
# ============================================================

df_treino.reset_index(drop=True).to_csv(
    arquivo_treino,
    index=False
)

df_teste.reset_index(drop=True).to_csv(
    arquivo_teste,
    index=False
)


# ============================================================
# VERIFICAÇÃO
# ============================================================

print("Total:", len(df))
print("Treino:", len(df_treino))
print("Teste:", len(df_teste))

print("\nPercentual de atrasos no treino:")
print(df_treino["DELAYED"].mean() * 100)

print("\nPercentual de atrasos no teste:")
print(df_teste["DELAYED"].mean() * 100)