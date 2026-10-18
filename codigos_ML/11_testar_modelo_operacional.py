import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    balanced_accuracy_score
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
    / "arquivo_teste_operacional.csv"
)

arquivo_treino = (
    pasta
    / "dataset_filtrado"
    / "arquivo_treino_operacional.csv"
)

arquivo_modelo = (
    pasta
    / "modelos"
    / "random_forest_operacional.joblib"
)

arquivo_calibrador = (
    pasta
    / "modelos"
    / "calibrador_operacional.joblib"
)

arquivo_comparacao = (
    pasta
    / "modelos"
    / "comparacao_real_vs_modelo.csv"
)


# ============================================================
# CONFIGURAÇÃO DO ALERTA
# ============================================================

LIMITE_ATRASO = 0.30


# ============================================================
# CARREGAR DADOS
# ============================================================

df_teste = pd.read_csv(
    arquivo_teste
)

df_treino = pd.read_csv(
    arquivo_treino
)

# ============================================================
# TARGET
# ============================================================

y_teste = (
    df_teste["DELAYED"]
    .astype(int)
)


# ============================================================
# FEATURES
# ============================================================

features_categoricas = [
    "AIRLINE",
    "DESTINATION_AIRPORT",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR"
]


features_numericas = [

    # Features antigas
    "DAY",
    "DEPARTURE_MINUTES",
    "SCHEDULED_TIME",
    "DISTANCE",

    # Features operacionais
    "CONGESTIONAMENTO_ATL",

    "QTD_VOOS_COMPANHIA_2H",
    "QTD_ATRASOS_COMPANHIA_2H",
    "TAXA_ATRASO_COMPANHIA_2H",

    "TEM_VOO_ANTERIOR",
    "ATRASO_SAIDA_VOO_ANTERIOR",
    "VOO_ANTERIOR_AINDA_NAO_SAIU"
]


features = (
    features_categoricas
    +
    features_numericas
)


X_teste = (
    df_teste[
        features
    ]
)


# ============================================================
# CARREGAR MODELO
# ============================================================

modelo = joblib.load(
    arquivo_modelo
)

calibrador = joblib.load(
    arquivo_calibrador
)


# ============================================================
# PROBABILIDADES BRUTAS
# ============================================================

prob_bruta = (
    modelo
    .predict_proba(
        X_teste
    )[:, 1]
)


# ============================================================
# PROBABILIDADES CALIBRADAS
# ============================================================

prob_calibrada = (
    calibrador
    .predict(
        prob_bruta
    )
)


# ============================================================
# MÉTRICAS - MODELO BRUTO
# ============================================================

roc_bruto = roc_auc_score(
    y_teste,
    prob_bruta
)

ap_bruto = average_precision_score(
    y_teste,
    prob_bruta
)

brier_bruto = brier_score_loss(
    y_teste,
    prob_bruta
)


# ============================================================
# MÉTRICAS - MODELO CALIBRADO
# ============================================================

roc_cal = roc_auc_score(
    y_teste,
    prob_calibrada
)

ap_cal = average_precision_score(
    y_teste,
    prob_calibrada
)

brier_cal = brier_score_loss(
    y_teste,
    prob_calibrada
)


# ============================================================
# BASELINE SIMPLES
# ============================================================
#
# Modelo que sempre responde a taxa média
# de atraso do conjunto de TREINO.
#
# ============================================================

taxa_treino = (
    df_treino["DELAYED"]
    .astype(int)
    .mean()
)


prob_baseline = np.full(
    len(y_teste),
    taxa_treino
)


brier_baseline = brier_score_loss(
    y_teste,
    prob_baseline
)


# ============================================================
# MELHORIA DO BRIER
# ============================================================

melhoria_brier = (
    (
        brier_baseline
        -
        brier_cal
    )
    /
    brier_baseline
    *
    100
)


# ============================================================
# DATAFRAME COM RESULTADOS
# ============================================================

resultado = df_teste[
    [
        "YEAR",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "FLIGHT_NUMBER",
        "DESTINATION_AIRPORT",
        "DELAYED"
    ]
].copy()


resultado[
    "PROB_BRUTA"
] = prob_bruta


resultado[
    "PROB_CALIBRADA"
] = prob_calibrada


# ============================================================
# REALIDADE x PREVISÃO COM LIMITE DE 30%
# ============================================================

resultado["REAL_ATRASO"] = (
    resultado["DELAYED"]
    .astype(int)
    .astype(bool)
)

resultado["MODELO_ATRASO"] = (
    resultado["PROB_CALIBRADA"]
    >= LIMITE_ATRASO
)

resultado["ACERTO"] = (
    resultado["REAL_ATRASO"]
    == resultado["MODELO_ATRASO"]
)


verdadeiro_positivo = (
    resultado["REAL_ATRASO"]
    & resultado["MODELO_ATRASO"]
).sum()

falso_positivo = (
    ~resultado["REAL_ATRASO"]
    & resultado["MODELO_ATRASO"]
).sum()

falso_negativo = (
    resultado["REAL_ATRASO"]
    & ~resultado["MODELO_ATRASO"]
).sum()

verdadeiro_negativo = (
    ~resultado["REAL_ATRASO"]
    & ~resultado["MODELO_ATRASO"]
).sum()

total = len(resultado)
acuracia = (
    (verdadeiro_positivo + verdadeiro_negativo) / total
    if total
    else 0
)

total_previstos_atrasados = verdadeiro_positivo + falso_positivo
precision = (
    verdadeiro_positivo / total_previstos_atrasados
    if total_previstos_atrasados
    else 0
)

total_atrasos_reais = verdadeiro_positivo + falso_negativo
recall = (
    verdadeiro_positivo / total_atrasos_reais
    if total_atrasos_reais
    else 0
)

f1 = f1_score(
    resultado["REAL_ATRASO"],
    resultado["MODELO_ATRASO"],
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    resultado["REAL_ATRASO"],
    resultado["MODELO_ATRASO"]
)

total_nao_atrasos_reais = verdadeiro_negativo + falso_positivo
especificidade = (
    verdadeiro_negativo / total_nao_atrasos_reais
    if total_nao_atrasos_reais
    else 0
)

total_previstos_sem_atraso = verdadeiro_negativo + falso_negativo
npv = (
    verdadeiro_negativo / total_previstos_sem_atraso
    if total_previstos_sem_atraso
    else 0
)


print("=" * 70)
print("RESULTADO DO MODELO")
print("=" * 70)
print()
print("MODELO BRUTO")
print(f"ROC AUC:           {roc_bruto:.6f}")
print(f"Average Precision: {ap_bruto:.6f}")
print(f"Brier Score:       {brier_bruto:.6f}")
print()
print("MODELO CALIBRADO")
print(f"ROC AUC:           {roc_cal:.6f}")
print(f"Average Precision: {ap_cal:.6f}")
print(f"Brier Score:       {brier_cal:.6f}")
print()
print("BASELINE")
print(f"Brier baseline:    {brier_baseline:.6f}")
print(f"Melhoria Brier:    {melhoria_brier:.2f}%")
print()
print(f"CLASSIFICAÇÃO COM LIMITE DE {LIMITE_ATRASO * 100:.0f}%")
print(f"Acurácia:          {acuracia * 100:.2f}%")
print(f"Precisão:          {precision * 100:.2f}%")
print(f"Recall:            {recall * 100:.2f}%")
print(f"F1-Score:          {f1 * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"Especificidade:    {especificidade * 100:.2f}%")
print(f"Negative Predictive Value (NPV): {npv * 100:.2f}%")
print()
print("=" * 70)
print("PRIMEIROS 100 VOOS")
print("=" * 70)

for _, linha in resultado.head(100).iterrows():
    print(
        f"{linha['AIRLINE']} "
        f"voo {int(linha['FLIGHT_NUMBER'])} "
        f"-> {linha['DESTINATION_AIRPORT']} | "
        f"REAL_ATRASO: {linha['REAL_ATRASO']} | "
        f"MODELO_ATRASO: {linha['MODELO_ATRASO']} | "
        f"PROBABILIDADE: {linha['PROB_CALIBRADA'] * 100:.1f}%"
    )


# ============================================================
# GRUPOS DE RISCO
# ============================================================

def classificar_risco(prob):

    if prob < 0.15:

        return "PROVAVELMENTE NO HORARIO"

    elif prob < 0.30:

        return "RISCO MODERADO"

    else:

        return "ALTO RISCO DE ATRASO"


resultado[
    "GRUPO_RISCO"
] = (
    resultado[
        "PROB_CALIBRADA"
    ]
    .apply(
        classificar_risco
    )
)


# ============================================================
# RESUMO DOS GRUPOS
# ============================================================

resumo_risco = (
    resultado
    .groupby(
        "GRUPO_RISCO"
    )
    .agg(

        quantidade=(
            "DELAYED",
            "size"
        ),

        probabilidade_media=(
            "PROB_CALIBRADA",
            "mean"
        ),

        taxa_real_atraso=(
            "DELAYED",
            "mean"
        ),

        quantidade_atrasos=(
            "DELAYED",
            "sum"
        )
    )
)


resumo_risco[
    "percentual_voos"
] = (
    resumo_risco[
        "quantidade"
    ]
    /
    len(resultado)
    *
    100
)


resumo_risco[
    "probabilidade_media"
] *= 100


resumo_risco[
    "taxa_real_atraso"
] *= 100


# ============================================================
# PERCENTUAL DE TODOS OS ATRASOS CAPTURADOS
# ============================================================

total_atrasos = (
    resultado[
        "DELAYED"
    ]
    .sum()
)


resumo_risco[
    "percentual_dos_atrasos"
] = (
    resumo_risco[
        "quantidade_atrasos"
    ]
    /
    total_atrasos
    *
    100
)


# ============================================================
# CALIBRAÇÃO POR FAIXAS
# ============================================================

bins = [
    0,
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


labels = [
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


resultado[
    "FAIXA_PROBABILIDADE"
] = pd.cut(
    resultado[
        "PROB_CALIBRADA"
    ],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)


calibracao_faixas = (
    resultado
    .groupby(
        "FAIXA_PROBABILIDADE",
        observed=False
    )
    .agg(

        quantidade=(
            "DELAYED",
            "size"
        ),

        probabilidade_media=(
            "PROB_CALIBRADA",
            "mean"
        ),

        taxa_real_atraso=(
            "DELAYED",
            "mean"
        )
    )
)


calibracao_faixas[
    "probabilidade_media"
] *= 100


calibracao_faixas[
    "taxa_real_atraso"
] *= 100


calibracao_faixas[
    "erro_calibracao"
] = (
    calibracao_faixas[
        "taxa_real_atraso"
    ]
    -
    calibracao_faixas[
        "probabilidade_media"
    ]
).abs()


# ============================================================
# PREVISÃO MÉDIA x RESULTADO REAL POR GRUPO DE RISCO
# ============================================================

resultado["GRUPO_RISCO_COMPARACAO"] = pd.cut(
    resultado["PROB_CALIBRADA"],
    bins=[0, 0.15, 0.30, 1.01],
    labels=[
        "PROVAVELMENTE NO HORARIO",
        "RISCO MODERADO",
        "ALTO RISCO DE ATRASO"
    ],
    include_lowest=True,
    right=False
)

resumo_comparacao = (
    resultado
    .groupby(
        "GRUPO_RISCO_COMPARACAO",
        observed=False
    )
    .agg(
        QUANTIDADE=("DELAYED", "size"),
        PROBABILIDADE_MEDIA=("PROB_CALIBRADA", "mean"),
        ATRASOS_REAIS=("DELAYED", "sum")
    )
)

resumo_comparacao["ATRASOS_ESPERADOS_MODELO"] = (
    resumo_comparacao["QUANTIDADE"]
    * resumo_comparacao["PROBABILIDADE_MEDIA"]
)

resumo_comparacao["PROBABILIDADE_MEDIA"] *= 100

# ============================================================
# SALVAR RESULTADOS
# ============================================================

pasta_modelos = (
    pasta
    / "modelos"
)


resultado.to_csv(
    pasta_modelos
    / "resultado_teste_operacional.csv",
    index=False
)


resumo_risco.to_csv(
    pasta_modelos
    / "resumo_risco_operacional.csv"
)


calibracao_faixas.to_csv(
    pasta_modelos
    / "calibracao_teste_operacional.csv"
)

colunas_comparacao = [
    "YEAR",
    "MONTH",
    "DAY",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "DESTINATION_AIRPORT",
    "PROB_CALIBRADA",
    "REAL_ATRASO",
    "MODELO_ATRASO",
    "ACERTO"
]

resultado[
    colunas_comparacao
].to_csv(
    arquivo_comparacao,
    index=False
)
