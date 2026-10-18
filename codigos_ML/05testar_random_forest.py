import pandas as pd
from pathlib import Path
from joblib import load

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
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

arquivo_modelo = (
    pasta
    / "modelos"
    / "modelo_random_forest.joblib"
)


# ============================================================
# CARREGAR MODELO
# ============================================================

modelo = load(
    arquivo_modelo
)


# ============================================================
# CARREGAR DATASET DE TESTE
# ============================================================

df_teste = pd.read_csv(
    arquivo_teste
)


# ============================================================
# X E y
# ============================================================

X_teste = df_teste.drop(
    columns=["DELAYED"]
)

y_teste = df_teste["DELAYED"]


# ============================================================
# PREVISÕES
# ============================================================

y_previsto = modelo.predict(
    X_teste
)

probabilidade_atraso = modelo.predict_proba(
    X_teste
)[:, 1]


# ============================================================
# MÉTRICAS
# ============================================================

accuracy = accuracy_score(
    y_teste,
    y_previsto
)

precision = precision_score(
    y_teste,
    y_previsto
)

recall = recall_score(
    y_teste,
    y_previsto
)

f1 = f1_score(
    y_teste,
    y_previsto
)

roc_auc = roc_auc_score(
    y_teste,
    probabilidade_atraso
)

pr_auc = average_precision_score(
    y_teste,
    probabilidade_atraso
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n==============================")
print("RESULTADOS RANDOM FOREST")
print("==============================")

print("\nACCURACY:")
print(accuracy)

print("\nPRECISION:")
print(precision)

print("\nRECALL:")
print(recall)

print("\nF1:")
print(f1)

print("\nROC AUC:")
print(roc_auc)

print("\nPR AUC:")
print(pr_auc)


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

print("\nMATRIZ DE CONFUSÃO:")

print(
    confusion_matrix(
        y_teste,
        y_previsto
    )
)


# ============================================================
# BASELINE DA PR AUC
# ============================================================

taxa_atrasos = y_teste.mean()

print("\nTAXA REAL DE ATRASOS:")
print(taxa_atrasos)

print("\nBASELINE PR AUC:")
print(taxa_atrasos)


# ============================================================
# EXEMPLOS
# ============================================================

resultado = X_teste.copy()

resultado["REAL"] = y_teste.values
resultado["PROB_ATRASO"] = probabilidade_atraso * 100
resultado["PREVISAO"] = y_previsto


print("\n==============================")
print("EXEMPLOS")
print("==============================")

print(
    resultado[
        [
            "AIRLINE",
            "DESTINATION_AIRPORT",
            "HOUR",
            "REAL",
            "PROB_ATRASO",
            "PREVISAO"
        ]
    ].head(20)
)