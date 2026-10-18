import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CAMINHOS
# ============================================================

pasta_filtrado = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado"
)

arquivo_voos = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso\dataset\atlanta_flights_com_horario_sem_cancelamento_periodo.csv"
)


# ============================================================
# CARREGAR DADOS
# ============================================================

df_destino = pd.read_csv(
    pasta_filtrado / "atraso_por_destino.csv"
)

df = pd.read_csv(arquivo_voos)


# ============================================================
# DESTINOS
# ============================================================

if df_destino["percentual_voos_atrasados"].max() <= 1:
    df_destino["percentual_voos_atrasados"] *= 100


# 15 destinos com mais voos
top_destinos = (
    df_destino
    .sort_values("numero_voos", ascending=False)
    .head(20)
)


# ============================================================
# DISTÂNCIA
# ============================================================

faixas_distancia = [
    0,
    500,
    1000,
    1500,
    2000,
    float("inf")
]

nomes_distancia = [
    "0-500",
    "500-1000",
    "1000-1500",
    "1500-2000",
    ">2000"
]

df["FAIXA_DISTANCIA"] = pd.cut(
    df["DISTANCE"],
    bins=faixas_distancia,
    labels=nomes_distancia,
    right=False
)

atraso_distancia = (
    df.groupby(
        "FAIXA_DISTANCIA",
        observed=True
    )
    .agg(
        numero_voos=("DELAYED", "size"),
        numero_atrasos=("DELAYED", "sum"),
        percentual_atrasos=("DELAYED", "mean")
    )
    .reset_index()
)

atraso_distancia["percentual_atrasos"] *= 100


# ============================================================
# DURAÇÃO PROGRAMADA
# ============================================================

faixas_tempo = [
    0,
    60,
    120,
    180,
    240,
    float("inf")
]

nomes_tempo = [
    "<1h",
    "1-2h",
    "2-3h",
    "3-4h",
    ">4h"
]

df["FAIXA_TEMPO"] = pd.cut(
    df["SCHEDULED_TIME"],
    bins=faixas_tempo,
    labels=nomes_tempo,
    right=False
)

atraso_tempo = (
    df.groupby(
        "FAIXA_TEMPO",
        observed=True
    )
    .agg(
        numero_voos=("DELAYED", "size"),
        numero_atrasos=("DELAYED", "sum"),
        percentual_atrasos=("DELAYED", "mean")
    )
    .reset_index()
)

atraso_tempo["percentual_atrasos"] *= 100


# ============================================================
# CRIAR LAYOUT
# ============================================================

fig = plt.figure(
    figsize=(16, 11),
    constrained_layout=True
)

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1, 1.2]
)

# Superior esquerdo
ax_distancia = fig.add_subplot(gs[0, 0])

# Superior direito
ax_tempo = fig.add_subplot(gs[0, 1])

# Parte inferior inteira
ax_destino = fig.add_subplot(gs[1, :])


# ============================================================
# GRÁFICO 1 - DISTÂNCIA
# ============================================================

barras = ax_distancia.bar(
    atraso_distancia["FAIXA_DISTANCIA"],
    atraso_distancia["percentual_atrasos"]
)

ax_distancia.set_title(
    "Delay percentage by flight distance"
)

ax_distancia.set_xlabel(
    "Distance"
)

ax_distancia.set_ylabel(
    "Delayed flights (%)"
)

maior = atraso_distancia[
    "percentual_atrasos"
].max()

ax_distancia.set_ylim(
    0,
    maior + 5
)

ax_distancia.grid(
    axis="y"
)

ax_distancia.bar_label(
    barras,
    fmt="%.1f%%",
    padding=3,
    color="black"
)


# ============================================================
# GRÁFICO 2 - DURAÇÃO
# ============================================================

barras = ax_tempo.bar(
    atraso_tempo["FAIXA_TEMPO"],
    atraso_tempo["percentual_atrasos"]
)

ax_tempo.set_title(
    "Delay percentage by scheduled flight time"
)

ax_tempo.set_xlabel(
    "Scheduled flight time"
)

ax_tempo.set_ylabel(
    "Delayed flights (%)"
)

maior = atraso_tempo[
    "percentual_atrasos"
].max()

ax_tempo.set_ylim(
    0,
    maior + 5
)

ax_tempo.grid(
    axis="y"
)

ax_tempo.bar_label(
    barras,
    fmt="%.1f%%",
    padding=3,
    color="black"
)


# ============================================================
# GRÁFICO 3 - TOP 15 DESTINOS
# ============================================================

barras = ax_destino.bar(
    top_destinos["DESTINATION_AIRPORT"],
    top_destinos["percentual_voos_atrasados"]
)

ax_destino.set_title(
    "Delay percentage for the 15 most frequent destinations"
)

ax_destino.set_xlabel(
    "Destination airport"
)

ax_destino.set_ylabel(
    "Delayed flights (%)"
)

maior = top_destinos[
    "percentual_voos_atrasados"
].max()

ax_destino.set_ylim(
    0,
    maior + 5
)

ax_destino.grid(
    axis="y"
)

ax_destino.bar_label(
    barras,
    fmt="%.1f%%",
    padding=3,
    color="black"
)


# ============================================================
# MOSTRAR
# ============================================================

plt.show()