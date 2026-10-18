import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ============================================================
# CAMINHOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
)

arquivo_treino = (
    pasta
    / "dataset_filtrado"
    / "arquivo_treino.csv"
)

arquivo_modelo = (
    pasta
    / "modelos"
    / "modelo_logistico.joblib"
)


# ============================================================
# CARREGAR TREINO
# ============================================================

df = pd.read_csv(arquivo_treino)


# ============================================================
# SEPARAR X E y
# ============================================================

X = df.drop(columns=["DELAYED"])

y = df["DELAYED"]


# ============================================================
# COLUNAS
# ============================================================

colunas_categoricas = [
    "AIRLINE",
    "DESTINATION_AIRPORT"
]

colunas_numericas = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "HOUR",
    "DEPARTURE_MINUTES",
    "SCHEDULED_TIME",
    "DISTANCE"
]


# ============================================================
# PREPARAR DADOS
# ============================================================

preparacao = ColumnTransformer(
    [
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            colunas_categoricas
        ),

        (
            "numericas",
            StandardScaler(),
            colunas_numericas
        )
    ]
)


# ============================================================
# CRIAR MODELO
# ============================================================

modelo = Pipeline(
    [
        ("preparacao", preparacao),

        ("modelo",LogisticRegression(max_iter=1000,class_weight="balanced"))
    ]
)


# ============================================================
# TREINAR
# ============================================================

modelo.fit(X, y)


# ============================================================
# SALVAR MODELO
# ============================================================

arquivo_modelo.parent.mkdir(
    parents=True,
    exist_ok=True
)

dump(modelo,arquivo_modelo)


print("Modelo treinado e salvo.")
print(arquivo_modelo)

preparacao_treinada = modelo.named_steps["preparacao"]
X_transformado = preparacao_treinada.transform(X.head(1))
print(X_transformado.toarray())