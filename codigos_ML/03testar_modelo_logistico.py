import pandas as pd
from pathlib import Path
from joblib import load

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
    / "modelo_logistico.joblib"
)


# ============================================================
# CARREGAR MODELO
# ============================================================

modelo = load(arquivo_modelo)


# ============================================================
# CARREGAR DATASET DE TESTE
# ============================================================

df_teste = pd.read_csv(arquivo_teste)


# ============================================================
# SEPARAR X E y
# ============================================================

X_teste = df_teste.drop(columns=["DELAYED"])

y_teste = df_teste["DELAYED"]


# ============================================================
# FAZER PREVISÕES
# ============================================================

y_previsto = modelo.predict(
    X_teste
)

probabilidade_atraso = modelo.predict_proba(
    X_teste
)[:, 1]


# ============================================================
# AVALIAR MODELO
# ============================================================

print("\nACCURACY:")
print(
    accuracy_score(
        y_teste,
        y_previsto
    )
)

print("\nPRECISION:")
print(
    precision_score(
        y_teste,
        y_previsto
    )
)

print("\nRECALL:")
print(
    recall_score(
        y_teste,
        y_previsto
    )
)

print("\nF1:")
print(
    f1_score(
        y_teste,
        y_previsto
    )
)

print("\nROC AUC:")
print(
    roc_auc_score(
        y_teste,
        probabilidade_atraso
    )
)

print("\nMATRIZ DE CONFUSÃO:")
print(
    confusion_matrix(
        y_teste,
        y_previsto
    )
)


# ============================================================
# MOSTRAR ALGUNS VOOS
# ============================================================

resultado = X_teste.copy()

resultado["REAL"] = y_teste.values

resultado["PROB_ATRASO"] = (
    probabilidade_atraso * 100
)

resultado["PREVISAO"] = y_previsto


print("\nEXEMPLOS:")

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