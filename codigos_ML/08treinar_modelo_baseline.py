import pandas as pd

from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss
)


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
    / "random_forest_natural.joblib"
)

arquivo_calibrador = (
    pasta
    / "modelos"
    / "calibrador_isotonico.joblib"
)


# ============================================================
# CARREGAR DADOS
# ============================================================

df = pd.read_csv(
    arquivo_treino
)


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

colunas_modelo = (
    colunas_categoricas
    + colunas_numericas
)


# ============================================================
# X E y
# ============================================================

X = df[colunas_modelo].copy()

y = df["DELAYED"].copy()


# ============================================================
# TREINO / VALIDAÇÃO
# ============================================================

X_treino, X_validacao, y_treino, y_validacao = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==============================")
print("DADOS")
print("==============================")

print("Treino:", len(X_treino))
print("Validação:", len(X_validacao))

print(
    f"Taxa atraso treino: "
    f"{y_treino.mean() * 100:.2f}%"
)

print(
    f"Taxa atraso validação: "
    f"{y_validacao.mean() * 100:.2f}%"
)


# ============================================================
# PREPARAÇÃO
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
# RANDOM FOREST NATURAL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)


modelo = Pipeline(
    [
        (
            "preparacao",
            preparacao
        ),

        (
            "modelo",
            random_forest
        )
    ]
)


# ============================================================
# TREINAR RANDOM FOREST
# ============================================================

print("\n==============================")
print("TREINAMENTO RANDOM FOREST")
print("==============================")

modelo.fit(
    X_treino,
    y_treino
)

print("Random Forest treinada.")


# ============================================================
# SCORE BRUTO NA VALIDAÇÃO
# ============================================================

prob_bruta = (
    modelo
    .predict_proba(X_validacao)[:, 1]
)


# ============================================================
# TREINAR CALIBRADOR
# ============================================================

print("\n==============================")
print("CALIBRAÇÃO")
print("==============================")

calibrador = IsotonicRegression(
    y_min=0,
    y_max=1,
    out_of_bounds="clip"
)

calibrador.fit(
    prob_bruta,
    y_validacao.astype(int)
)


# ============================================================
# PROBABILIDADE CALIBRADA
# ============================================================

prob_calibrada = calibrador.predict(
    prob_bruta
)


# ============================================================
# MÉTRICAS ANTES
# ============================================================

ap_bruta = average_precision_score(
    y_validacao,
    prob_bruta
)

roc_bruta = roc_auc_score(
    y_validacao,
    prob_bruta
)

brier_bruta = brier_score_loss(
    y_validacao,
    prob_bruta
)


# ============================================================
# MÉTRICAS DEPOIS
# ============================================================

ap_calibrada = average_precision_score(
    y_validacao,
    prob_calibrada
)

roc_calibrada = roc_auc_score(
    y_validacao,
    prob_calibrada
)

brier_calibrada = brier_score_loss(
    y_validacao,
    prob_calibrada
)


# ============================================================
# COMPARAÇÃO
# ============================================================

print("\n==============================")
print("ANTES DA CALIBRAÇÃO")
print("==============================")

print(
    f"Average Precision: {ap_bruta:.6f}"
)

print(
    f"ROC AUC:           {roc_bruta:.6f}"
)

print(
    f"Brier Score:       {brier_bruta:.6f}"
)


print("\n==============================")
print("DEPOIS DA CALIBRAÇÃO")
print("==============================")

print(
    f"Average Precision: {ap_calibrada:.6f}"
)

print(
    f"ROC AUC:           {roc_calibrada:.6f}"
)

print(
    f"Brier Score:       {brier_calibrada:.6f}"
)


# ============================================================
# TABELA DE CALIBRAÇÃO
# ============================================================

resultado = pd.DataFrame(
    {
        "REAL": y_validacao.values,
        "PROB_BRUTA": prob_bruta,
        "PROB_CALIBRADA": prob_calibrada
    }
)


# ============================================================
# FAIXAS
# ============================================================

limites = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    1.01
]

nomes_faixas = [
    "0-5%",
    "5-10%",
    "10-15%",
    "15-20%",
    "20-25%",
    "25-30%",
    "30-40%",
    "40-50%",
    ">50%"
]


resultado["FAIXA"] = pd.cut(
    resultado["PROB_CALIBRADA"],
    bins=limites,
    labels=nomes_faixas,
    right=False,
    include_lowest=True
)


# ============================================================
# AGRUPAR
# ============================================================

tabela = (
    resultado
    .groupby(
        "FAIXA",
        observed=False
    )
    .agg(
        QUANTIDADE=(
            "REAL",
            "size"
        ),

        PROB_MEDIA=(
            "PROB_CALIBRADA",
            "mean"
        ),

        ATRASO_REAL=(
            "REAL",
            "mean"
        )
    )
    .reset_index()
)


tabela["PROB_MEDIA"] *= 100
tabela["ATRASO_REAL"] *= 100


# ============================================================
# MOSTRAR TABELA
# ============================================================

print("\n==============================")
print("PROBABILIDADE CALIBRADA")
print("==============================")

print(
    tabela.to_string(
        index=False,

        formatters={
            "PROB_MEDIA":
                lambda x: (
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "-"
                ),

            "ATRASO_REAL":
                lambda x: (
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "-"
                )
        }
    )
)


# ============================================================
# SALVAR
# ============================================================

arquivo_modelo.parent.mkdir(
    parents=True,
    exist_ok=True
)

dump(
    modelo,
    arquivo_modelo
)

dump(
    calibrador,
    arquivo_calibrador
)


print("\n==============================")
print("SALVO")
print("==============================")

print("\nRandom Forest:")
print(arquivo_modelo)

print("\nCalibrador:")
print(arquivo_calibrador)