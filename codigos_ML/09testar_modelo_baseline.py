import pandas as pd  # type: ignore[reportMissingModuleSource]
import numpy as np

from pathlib import Path
from joblib import load

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

arquivo_teste = (
    pasta
    / "dataset_filtrado"
    / "arquivo_teste.csv"
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
# COLUNAS
# ============================================================

colunas_modelo = [
    "AIRLINE",
    "DESTINATION_AIRPORT",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR",
    "DAY",
    "DEPARTURE_MINUTES",
    "SCHEDULED_TIME",
    "DISTANCE"
]


# ============================================================
# CARREGAR MODELO E DADOS
# ============================================================

modelo = load(arquivo_modelo)
calibrador = load(arquivo_calibrador)

df = pd.read_csv(arquivo_teste)

X = df[colunas_modelo].copy()
y = df["DELAYED"].astype(int).copy()


# ============================================================
# PROBABILIDADE DE ATRASO
# ============================================================

prob_bruta = modelo.predict_proba(X)[:, 1]

prob_atraso = calibrador.predict(
    prob_bruta
)


# ============================================================
# MÉTRICAS GERAIS
# ============================================================

taxa_atraso = y.mean()

average_precision = average_precision_score(
    y,
    prob_atraso
)

roc_auc = roc_auc_score(
    y,
    prob_atraso
)

brier = brier_score_loss(
    y,
    prob_atraso
)


# Baseline:
# prever 17.9% para todos os voos
prob_baseline = np.full(
    len(y),
    taxa_atraso
)

brier_baseline = brier_score_loss(
    y,
    prob_baseline
)


# ============================================================
# CRIAR RESULTADO
# ============================================================

resultado = X.copy()

resultado["REAL"] = y.values
resultado["PROB_ATRASO"] = prob_atraso


# ============================================================
# CLASSIFICAR EM 3 NÍVEIS DE RISCO
# ============================================================

def classificar_risco(prob):

    if prob < 0.15:
        return "PROVAVELMENTE NO HORARIO"

    elif prob < 0.30:
        return "RISCO MODERADO"

    else:
        return "ALTO RISCO DE ATRASO"


resultado["RISCO"] = (
    resultado["PROB_ATRASO"]
    .apply(classificar_risco)
)


# ============================================================
# RESUMO POR GRUPO
# ============================================================

ordem = [
    "PROVAVELMENTE NO HORARIO",
    "RISCO MODERADO",
    "ALTO RISCO DE ATRASO"
]


resumo = (
    resultado
    .groupby("RISCO")
    .agg(
        VOOS=(
            "REAL",
            "size"
        ),

        PROB_MEDIA=(
            "PROB_ATRASO",
            "mean"
        ),

        ATRASO_REAL=(
            "REAL",
            "mean"
        ),

        TOTAL_ATRASADOS=(
            "REAL",
            "sum"
        )
    )
    .reindex(ordem)
)


resumo["PERCENTUAL_DATASET"] = (
    resumo["VOOS"]
    / len(resultado)
)

resumo["PROB_MEDIA"] *= 100
resumo["ATRASO_REAL"] *= 100
resumo["PERCENTUAL_DATASET"] *= 100


resumo["ERRO_CALIBRACAO"] = abs(
    resumo["PROB_MEDIA"]
    - resumo["ATRASO_REAL"]
)


# ============================================================
# MÉTRICAS ESPECÍFICAS DAS FAIXAS
# ============================================================

baixo = (
    resultado["RISCO"]
    == "PROVAVELMENTE NO HORARIO"
)

moderado = (
    resultado["RISCO"]
    == "RISCO MODERADO"
)

alto = (
    resultado["RISCO"]
    == "ALTO RISCO DE ATRASO"
)


# ------------------------------------------------------------
# BAIXO RISCO
# ------------------------------------------------------------

taxa_atraso_baixo = (
    resultado.loc[baixo, "REAL"]
    .mean()
)

taxa_no_horario_baixo = (
    1 - taxa_atraso_baixo
)


# ------------------------------------------------------------
# MODERADO
# ------------------------------------------------------------

taxa_atraso_moderado = (
    resultado.loc[moderado, "REAL"]
    .mean()
)


# ------------------------------------------------------------
# ALTO RISCO
# ------------------------------------------------------------

taxa_atraso_alto = (
    resultado.loc[alto, "REAL"]
    .mean()
)


# Quantos de TODOS os atrasos estão no grupo de alto risco
atrasos_alto = (
    resultado.loc[alto, "REAL"]
    .sum()
)

total_atrasos = y.sum()

recall_alto_risco = (
    atrasos_alto
    / total_atrasos
)


# ============================================================
# SEPARAÇÃO ENTRE ALTO E BAIXO RISCO
# ============================================================

razao_risco = (
    taxa_atraso_alto
    / taxa_atraso_baixo
)


# ============================================================
# ERRO MÉDIO DE CALIBRAÇÃO
# ============================================================

erro_calibracao = (
    resumo["ERRO_CALIBRACAO"]
    * resumo["VOOS"]
).sum() / resumo["VOOS"].sum()


# ============================================================
# PRINT 1 - VISÃO GERAL
# ============================================================

print("\n")
print("=" * 65)
print("                 RESULTADO GERAL")
print("=" * 65)

print(f"\nVoos analisados:       {len(y):,}")
print(f"Voos atrasados:        {int(y.sum()):,}")
print(f"Taxa média de atraso:  {taxa_atraso * 100:.2f}%")


print("\nMétricas gerais:")

print(
    f"Average Precision:     "
    f"{average_precision:.3f}"
)

print(
    f"Baseline AP:            "
    f"{taxa_atraso:.3f}"
)

print(
    f"ROC AUC:                "
    f"{roc_auc:.3f}"
)

print(
    f"Brier Score:            "
    f"{brier:.3f}"
)

print(
    f"Brier baseline:         "
    f"{brier_baseline:.3f}"
)


# ============================================================
# PRINT 2 - 3 NÍVEIS DE RISCO
# ============================================================

print("\n")
print("=" * 80)
print("                    CLASSIFICAÇÃO DE RISCO")
print("=" * 80)

print(
    f"\n{'GRUPO':<28}"
    f"{'VOOS':>9}"
    f"{'% BASE':>10}"
    f"{'ESTIMADO':>12}"
    f"{'REAL':>10}"
    f"{'ERRO':>10}"
)

print("-" * 80)


for nome, linha in resumo.iterrows():

    print(
        f"{nome:<28}"
        f"{int(linha['VOOS']):>9,}"
        f"{linha['PERCENTUAL_DATASET']:>9.1f}%"
        f"{linha['PROB_MEDIA']:>11.1f}%"
        f"{linha['ATRASO_REAL']:>9.1f}%"
        f"{linha['ERRO_CALIBRACAO']:>8.1f} pp"
    )


# ============================================================
# PRINT 3 - INTERPRETAÇÃO
# ============================================================

print("\n")
print("=" * 65)
print("               ROBUSTEZ DO SISTEMA")
print("=" * 65)


print("\nPROVAVELMENTE NO HORARIO")

print(
    f"→ {taxa_no_horario_baixo * 100:.1f}% "
    f"realmente saíram no horário"
)

print(
    f"→ {taxa_atraso_baixo * 100:.1f}% "
    f"ainda atrasaram"
)


print("\nRISCO MODERADO")

print(
    f"→ {taxa_atraso_moderado * 100:.1f}% "
    f"realmente atrasaram"
)


print("\nALTO RISCO")

print(
    f"→ {taxa_atraso_alto * 100:.1f}% "
    f"realmente atrasaram"
)

print(
    f"→ esse grupo contém "
    f"{recall_alto_risco * 100:.1f}% "
    f"de todos os atrasos da base"
)


print("\nSEPARAÇÃO DE RISCO")

print(
    f"→ um voo classificado como ALTO RISCO "
    f"atrasa {razao_risco:.1f}x mais "
    f"que um voo classificado como baixo risco"
)


print("\nCALIBRAÇÃO")

print(
    f"→ erro médio entre probabilidade estimada "
    f"e taxa real: {erro_calibracao:.2f} pontos percentuais"
)


# ============================================================
# PRINT 4 - EXEMPLOS INDIVIDUAIS
# ============================================================

resultado_exemplos = resultado[
    [
        "AIRLINE",
        "DESTINATION_AIRPORT",
        "HOUR",
        "PROB_ATRASO",
        "RISCO",
        "REAL"
    ]
].copy()


resultado_exemplos["PROB_ATRASO"] *= 100

resultado_exemplos["REAL"] = (
    resultado_exemplos["REAL"]
    .map({
        0: "NO HORARIO",
        1: "ATRASOU"
    })
)


print("\n")
print("=" * 95)
print("                         EXEMPLOS DE VOOS")
print("=" * 95)


print(
    resultado_exemplos
    .head(20)
    .to_string(
        index=False,
        formatters={
            "PROB_ATRASO":
                lambda x: f"{x:.1f}%"
        }
    )
)