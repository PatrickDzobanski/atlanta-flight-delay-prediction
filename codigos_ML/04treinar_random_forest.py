import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


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
    / "modelo_random_forest.joblib"
)


# ============================================================
# CARREGAR DATASET DE TREINO
# ============================================================

df = pd.read_csv(
    arquivo_treino
)


# ============================================================
# X E y
# ============================================================

X = df.drop(
    columns=["DELAYED"]
)

y = df["DELAYED"]


# ============================================================
# COLUNAS
# ============================================================

colunas_categoricas = [
    "AIRLINE",
    "DESTINATION_AIRPORT",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR"
]

colunas_numericas = [
    "DAY",
    "DEPARTURE_MINUTES",
    "SCHEDULED_TIME",
    "DISTANCE"
]


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

preparacao = ColumnTransformer(
    [
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            colunas_categoricas
        ),

        (
            "numericas",
            "passthrough",
            colunas_numericas
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# PIPELINE
# ============================================================

modelo = Pipeline(
    [
        ("preparacao", preparacao),
        ("modelo", random_forest)
    ]
)


# ============================================================
# TREINAR
# ============================================================

print("Treinando Random Forest...")

modelo.fit(
    X,
    y
)

print("Treinamento concluído.")


# ============================================================
# CRIAR PASTA, CASO NECESSÁRIO
# ============================================================

arquivo_modelo.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SALVAR MODELO
# ============================================================

dump(
    modelo,
    arquivo_modelo
)


print("\nModelo salvo em:")
print(arquivo_modelo)