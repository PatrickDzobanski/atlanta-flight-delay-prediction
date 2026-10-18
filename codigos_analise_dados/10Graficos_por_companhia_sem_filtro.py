import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PASTA DOS CSVs PROCESSADOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado"
)


# ============================================================
# CARREGAR OS CSVs
# ============================================================

df_companhia = pd.read_csv(pasta / "resumo_atraso.csv")
df_hora = pd.read_csv(pasta / "atraso_hora_companhia.csv")
df_periodo = pd.read_csv(pasta / "atraso_periodo_companhia.csv")
df_dia = pd.read_csv(pasta / "atraso_dia_companhia.csv")


# ============================================================
# GARANTIR QUE AS PORCENTAGENS ESTEJAM ENTRE 0 E 100
# ============================================================

if df_companhia["percentual_atrasos"].max() <= 1:
    df_companhia["percentual_atrasos"] *= 100

if df_hora["percentual_atrasos"].max() <= 1:
    df_hora["percentual_atrasos"] *= 100

if df_periodo["percentual_atrasos"].max() <= 1:
    df_periodo["percentual_atrasos"] *= 100

if df_dia["percentual_atrasos"].max() <= 1:
    df_dia["percentual_atrasos"] *= 100


# ============================================================
# PEGAR AS 10 COMPANHIAS COM MAIS VOOS
# ============================================================

top_10 = (
    df_companhia
    .sort_values("total_voos", ascending=False)
    .head(10)["AIRLINE"]
    .tolist()
)

print("10 companhias com mais voos:")
print(top_10)


# ============================================================
# FILTRAR OS OUTROS DATAFRAMES
# ============================================================

companhias_top = df_companhia[
    df_companhia["AIRLINE"].isin(top_10)
].copy()

hora_top = df_hora[
    df_hora["AIRLINE"].isin(top_10)
].copy()

periodo_top = df_periodo[
    df_periodo["AIRLINE"].isin(top_10)
].copy()

dia_top = df_dia[
    df_dia["AIRLINE"].isin(top_10)
].copy()


# ============================================================
# CRIAR FIGURA 2 x 2
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))


# ============================================================
# GRÁFICO 1 - ATRASO POR COMPANHIA
# ============================================================

companhias_top = companhias_top.sort_values(
    "percentual_atrasos",
    ascending=False
)

barras = axes[0, 0].bar(
    companhias_top["AIRLINE"],
    companhias_top["percentual_atrasos"]
)

axes[0, 0].bar_label(
    barras,
    fmt="%.1f%%",
    padding=3,
    color="black"
)

axes[0, 0].set_title("Delay percentage by airline")
axes[0, 0].set_xlabel("Airline")
axes[0, 0].set_ylabel("Delayed flights (%)")

maior_valor = companhias_top["percentual_atrasos"].max()
axes[0, 0].set_ylim(0, maior_valor + 5)

axes[0, 0].grid(axis="y")


# ============================================================
# GRÁFICO 2 - ATRASO POR HORA + COMPANHIA
# ============================================================

for companhia in top_10:

    dados = hora_top[
        hora_top["AIRLINE"] == companhia
    ]

    axes[0, 1].plot(
        dados["HOUR"],
        dados["percentual_atrasos"],
        marker="o",
        label=companhia
    )

axes[0, 1].set_title("Delay percentage by hour and airline")
axes[0, 1].set_xlabel("Hour")
axes[0, 1].set_ylabel("Delayed flights (%)")

maior_valor = hora_top["percentual_atrasos"].max()
axes[0, 1].set_ylim(0, maior_valor + 5)

axes[0, 1].set_xticks(range(5, 24))
axes[0, 1].grid()

axes[0, 1].legend(fontsize=8)


# ============================================================
# GRÁFICO 3 - ATRASO POR PERÍODO + COMPANHIA
# ============================================================

ordem_periodos = [
    "Madrugada",
    "Manhã",
    "Tarde",
    "Noite"
]

for companhia in top_10:

    dados = periodo_top[
        periodo_top["AIRLINE"] == companhia
    ]

    dados = (
        dados
        .set_index("PERIODO")
        .reindex(ordem_periodos)
        .reset_index()
    )

    axes[1, 0].plot(
        dados["PERIODO"],
        dados["percentual_atrasos"],
        marker="o",
        label=companhia
    )

axes[1, 0].set_title("Delay percentage by period and airline")
axes[1, 0].set_xlabel("Period of day")
axes[1, 0].set_ylabel("Delayed flights (%)")

maior_valor = periodo_top["percentual_atrasos"].max()
axes[1, 0].set_ylim(0, maior_valor + 5)

axes[1, 0].grid()

axes[1, 0].legend(fontsize=8)


# ============================================================
# GRÁFICO 4 - ATRASO POR DIA DA SEMANA + COMPANHIA
# ============================================================

for companhia in top_10:

    dados = dia_top[
        dia_top["AIRLINE"] == companhia
    ]

    axes[1, 1].plot(
        dados["DAY_OF_WEEK"],
        dados["percentual_atrasos"],
        marker="o",
        label=companhia
    )

axes[1, 1].set_title("Delay percentage by day and airline")
axes[1, 1].set_xlabel("Day of week")
axes[1, 1].set_ylabel("Delayed flights (%)")

maior_valor = dia_top["percentual_atrasos"].max()
axes[1, 1].set_ylim(0, maior_valor + 5)

axes[1, 1].set_xticks(range(1, 8))
axes[1, 1].grid()

axes[1, 1].legend(fontsize=8)


# ============================================================
# ORGANIZAR E MOSTRAR
# ============================================================

plt.tight_layout()
plt.show()