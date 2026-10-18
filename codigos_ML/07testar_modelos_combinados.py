import pandas as pd
import json

from pathlib import Path
from joblib import load

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)


# ============================================================
# CAMINHOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
)

arquivo_teste = (
    pasta
    / "dataset_filtrado"
    / "arquivo_teste.csv"
)

arquivo_modelo_balanceado = (
    pasta
    / "modelos"
    / "random_forest_balanceado.joblib"
)

arquivo_modelo_natural = (
    pasta
    / "modelos"
    / "random_forest_natural.joblib"
)

arquivo_pesos = (
    pasta
    / "modelos"
    / "pesos_combinacao.json"
)


# ============================================================
# COLUNAS UTILIZADAS PELOS MODELOS
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
# CARREGAR MODELOS
# ============================================================

modelo_balanceado = load(
    arquivo_modelo_balanceado
)

modelo_natural = load(
    arquivo_modelo_natural
)


# ============================================================
# CARREGAR PESOS
# ============================================================

with open(
    arquivo_pesos,
    "r"
) as arquivo:

    pesos = json.load(arquivo)


peso_balanceado = pesos["peso_balanceado"]
peso_natural = pesos["peso_natural"]


# ============================================================
# CARREGAR DATASET DE TESTE
# ============================================================

df = pd.read_csv(
    arquivo_teste
)


# ============================================================
# X E y
# ============================================================

X = df[colunas_modelo].copy()

y = df["DELAYED"].copy()


# ============================================================
# PROBABILIDADES DOS DOIS MODELOS
# ============================================================

prob_balanceado = (
    modelo_balanceado
    .predict_proba(X)[:, 1]
)

prob_natural = (
    modelo_natural
    .predict_proba(X)[:, 1]
)


# ============================================================
# SCORE COMBINADO
# ============================================================

prob_combinada = (
    peso_balanceado
    * prob_balanceado

    +

    peso_natural
    * prob_natural
)


# ============================================================
# DESACORDO ENTRE OS MODELOS
# ============================================================

desacordo = abs(
    prob_balanceado
    - prob_natural
)


# ============================================================
# AVERAGE PRECISION
# ============================================================

ap_balanceado = average_precision_score(
    y,
    prob_balanceado
)

ap_natural = average_precision_score(
    y,
    prob_natural
)

ap_combinado = average_precision_score(
    y,
    prob_combinada
)


# ============================================================
# ROC AUC
# ============================================================

roc_balanceado = roc_auc_score(
    y,
    prob_balanceado
)

roc_natural = roc_auc_score(
    y,
    prob_natural
)

roc_combinado = roc_auc_score(
    y,
    prob_combinada
)


# ============================================================
# BASELINE
# ============================================================

baseline = y.mean()


# ============================================================
# RESULTADOS PRINCIPAIS
# ============================================================

print("\n==============================")
print("RESULTADOS NO DATASET DE TESTE")
print("==============================")

print("\nTotal de voos:")
print(len(y))

print("\nTaxa real de atraso:")
print(f"{baseline * 100:.2f} %")


print("\n==============================")
print("PESOS UTILIZADOS")
print("==============================")

print(
    f"Modelo balanceado: "
    f"{peso_balanceado:.2f}"
)

print(
    f"Modelo natural:     "
    f"{peso_natural:.2f}"
)


print("\n==============================")
print("AVERAGE PRECISION")
print("==============================")

print(
    f"Baseline:    {baseline:.6f}"
)

print(
    f"Balanceado:  {ap_balanceado:.6f}"
)

print(
    f"Natural:     {ap_natural:.6f}"
)

print(
    f"Combinado:   {ap_combinado:.6f}"
)


print("\n==============================")
print("ROC AUC")
print("==============================")

print(
    f"Balanceado:  {roc_balanceado:.6f}"
)

print(
    f"Natural:     {roc_natural:.6f}"
)

print(
    f"Combinado:   {roc_combinado:.6f}"
)


# ============================================================
# VERIFICAR MELHORIA
# ============================================================

melhor_individual = max(
    ap_balanceado,
    ap_natural
)

melhoria = (
    ap_combinado
    - melhor_individual
)


print("\n==============================")
print("GANHO DA COMBINAÇÃO")
print("==============================")

print(
    f"Melhor modelo individual: "
    f"{melhor_individual:.6f}"
)

print(
    f"Modelo combinado:         "
    f"{ap_combinado:.6f}"
)

print(
    f"Diferença:                "
    f"{melhoria:+.6f}"
)


# ============================================================
# CRIAR DATAFRAME DE RESULTADOS
# ============================================================

resultado = X.copy()

resultado["REAL"] = y.values

resultado["PROB_BALANCEADO"] = (
    prob_balanceado * 100
)

resultado["PROB_NATURAL"] = (
    prob_natural * 100
)

resultado["SCORE_COMBINADO"] = (
    prob_combinada * 100
)

resultado["DESACORDO"] = (
    desacordo * 100
)


# ============================================================
# EXEMPLOS
# ============================================================

print("\n==============================")
print("EXEMPLOS")
print("==============================")

exemplos = resultado[
    [
        "AIRLINE",
        "DESTINATION_AIRPORT",
        "HOUR",
        "REAL",
        "PROB_BALANCEADO",
        "PROB_NATURAL",
        "SCORE_COMBINADO",
        "DESACORDO"
    ]
].head(20)

print(
    exemplos.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# DISTRIBUIÇÃO DO SCORE
# ============================================================

print("\n==============================")
print("DISTRIBUIÇÃO DO SCORE")
print("==============================")

print(
    resultado["SCORE_COMBINADO"]
    .describe()
)


# ============================================================
# DISTRIBUIÇÃO DO DESACORDO
# ============================================================

print("\n==============================")
print("DISTRIBUIÇÃO DO DESACORDO")
print("==============================")

print(
    resultado["DESACORDO"]
    .describe()
)