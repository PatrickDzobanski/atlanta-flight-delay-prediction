import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado"
)

# Fator de suavização
k = 50


# ============================================================
# CARREGAR CSVs
# ============================================================

# Resumo geral por companhia
df_companhia = pd.read_csv(
    pasta / "resumo_atraso.csv"
)

# Companhia + variável
df_hora = pd.read_csv(
    pasta / "atraso_hora_companhia.csv"
)

df_periodo = pd.read_csv(
    pasta / "atraso_periodo_companhia.csv"
)

df_dia = pd.read_csv(
    pasta / "atraso_dia_companhia.csv"
)


# Taxas-base sem separar por companhia
taxa_base_hora = pd.read_csv(
    pasta / "atraso_por_hora.csv"
)

taxa_base_periodo = pd.read_csv(
    pasta / "atraso_por_periodo.csv"
)

taxa_base_dia = pd.read_csv(
    pasta / "atraso_por_dia.csv"
)


# ============================================================
# FUNÇÃO PARA CONVERTER PORCENTAGEM PARA PROPORÇÃO
# ============================================================

def para_proporcao(serie):

    # Se estiver como 18, 25, 30...
    # transforma para 0.18, 0.25, 0.30...
    if serie.max() > 1:
        return serie / 100

    # Se já estiver entre 0 e 1, não faz nada
    return serie


# ============================================================
# PREPARAR TAXAS BRUTAS
# ============================================================

df_companhia["taxa_bruta"] = para_proporcao(
    df_companhia["percentual_atrasos"]
)

df_hora["taxa_bruta"] = para_proporcao(
    df_hora["percentual_atrasos"]
)

df_periodo["taxa_bruta"] = para_proporcao(
    df_periodo["percentual_atrasos"]
)

df_dia["taxa_bruta"] = para_proporcao(
    df_dia["percentual_atrasos"]
)


# ============================================================
# TAXA GERAL DE ATRASO
# ============================================================

taxa_geral = (
    df_companhia["total_atrasos"].sum()
    /
    df_companhia["total_voos"].sum()
)

print(
    "Taxa geral de atraso:",
    round(taxa_geral * 100, 2),
    "%"
)


# ============================================================
# 1 - TAXA AJUSTADA POR COMPANHIA
# ============================================================

df_companhia["taxa_ajustada"] = (
    df_companhia["total_atrasos"]
    + k * taxa_geral
) / (
    df_companhia["total_voos"] + k
)

# Converter para %
df_companhia["taxa_ajustada"] *= 100


# ============================================================
# 2 - TAXA AJUSTADA POR HORA + COMPANHIA
# ============================================================

# Taxa geral de atraso de cada hora
taxa_base_hora["taxa_base"] = para_proporcao(
    taxa_base_hora["DELAYED"]
)

# Adicionar a taxa-base de cada hora ao DataFrame
df_hora = df_hora.merge(
    taxa_base_hora[["HOUR", "taxa_base"]],
    on="HOUR",
    how="left"
)

# Reconstruir quantidade de atrasos
df_hora["num_atrasos"] = (
    df_hora["taxa_bruta"]
    * df_hora["num_voos"]
)

# Taxa suavizada
df_hora["taxa_ajustada"] = (
    df_hora["num_atrasos"]
    + k * df_hora["taxa_base"]
) / (
    df_hora["num_voos"] + k
)

df_hora["taxa_ajustada"] *= 100


# ============================================================
# 3 - TAXA AJUSTADA POR PERÍODO + COMPANHIA
# ============================================================

taxa_base_periodo["taxa_base"] = para_proporcao(
    taxa_base_periodo["DELAYED"]
)

df_periodo = df_periodo.merge(
    taxa_base_periodo[["PERIODO", "taxa_base"]],
    on="PERIODO",
    how="left"
)

df_periodo["num_atrasos"] = (
    df_periodo["taxa_bruta"]
    * df_periodo["num_voos"]
)

df_periodo["taxa_ajustada"] = (
    df_periodo["num_atrasos"]
    + k * df_periodo["taxa_base"]
) / (
    df_periodo["num_voos"] + k
)

df_periodo["taxa_ajustada"] *= 100


# ============================================================
# 4 - TAXA AJUSTADA POR DIA + COMPANHIA
# ============================================================

taxa_base_dia["taxa_base"] = para_proporcao(
    taxa_base_dia["DELAYED"]
)

df_dia = df_dia.merge(
    taxa_base_dia[["DAY_OF_WEEK", "taxa_base"]],
    on="DAY_OF_WEEK",
    how="left"
)

df_dia["num_atrasos"] = (
    df_dia["taxa_bruta"]
    * df_dia["num_voos"]
)

df_dia["taxa_ajustada"] = (
    df_dia["num_atrasos"]
    + k * df_dia["taxa_base"]
) / (
    df_dia["num_voos"] + k
)

df_dia["taxa_ajustada"] *= 100


# ============================================================
# DIAGNÓSTICO
# ============================================================

print("\nQuantidade de voos por combinação HORA + COMPANHIA:")

print(
    df_hora["num_voos"].describe()
)


print("\n20 maiores taxas ajustadas por hora:")

print(
    df_hora[
        [
            "HOUR",
            "AIRLINE",
            "num_voos",
            "taxa_bruta",
            "taxa_base",
            "taxa_ajustada"
        ]
    ]
    .sort_values(
        "taxa_ajustada",
        ascending=False
    )
    .head(20)
)


# ============================================================
# PEGAR AS 10 COMPANHIAS COM MAIS VOOS
# ============================================================

top_10 = (
    df_companhia
    .sort_values(
        "total_voos",
        ascending=False
    )
    .head(10)["AIRLINE"]
    .tolist()
)

print("\n10 companhias com mais voos:")
print(top_10)


# ============================================================
# FILTRAR PARA AS TOP 10
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

fig, axes = plt.subplots(
    2,
    2,
    figsize=(20, 12),
    constrained_layout=True
)


# ============================================================
# GRÁFICO 1 - COMPANHIA
# ============================================================

companhias_top = companhias_top.sort_values(
    "taxa_ajustada",
    ascending=False
)

barras = axes[0, 0].bar(
    companhias_top["AIRLINE"],
    companhias_top["taxa_ajustada"]
)

axes[0, 0].set_title(
    "Adjusted delay rate by airline"
)

axes[0, 0].set_xlabel("Airline")

axes[0, 0].set_ylabel(
    "Adjusted delayed flights (%)"
)

maior = companhias_top[
    "taxa_ajustada"
].max()

axes[0, 0].set_ylim(
    0,
    maior + 5
)

axes[0, 0].grid(
    axis="y"
)

axes[0, 0].bar_label(
    barras,
    fmt="%.1f%%",
    padding=3
)


# ============================================================
# GRÁFICO 2 - HORA + COMPANHIA
# ============================================================

for companhia in top_10:

    dados = (
        hora_top[
            hora_top["AIRLINE"] == companhia
        ]
        .sort_values("HOUR")
    )

    axes[0, 1].plot(
        dados["HOUR"],
        dados["taxa_ajustada"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label=companhia
    )


axes[0, 1].set_title(
    "Adjusted delay rate by hour and airline"
)

axes[0, 1].set_xlabel("Hour")

axes[0, 1].set_ylabel(
    "Adjusted delayed flights (%)"
)

maior = hora_top[
    "taxa_ajustada"
].max()

axes[0, 1].set_ylim(
    0,
    maior + 5
)

axes[0, 1].set_xticks(
    range(5, 24)
)

axes[0, 1].grid()

axes[0, 1].legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)


# ============================================================
# GRÁFICO 3 - PERÍODO + COMPANHIA
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
    ].copy()

    dados = (
        dados
        .set_index("PERIODO")
        .reindex(ordem_periodos)
        .reset_index()
    )

    axes[1, 0].plot(
        dados["PERIODO"],
        dados["taxa_ajustada"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=companhia
    )


axes[1, 0].set_title(
    "Adjusted delay rate by period and airline"
)

axes[1, 0].set_xlabel(
    "Period of day"
)

axes[1, 0].set_ylabel(
    "Adjusted delayed flights (%)"
)

maior = periodo_top[
    "taxa_ajustada"
].max()

axes[1, 0].set_ylim(
    0,
    maior + 5
)

axes[1, 0].grid()

axes[1, 0].legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)


# ============================================================
# GRÁFICO 4 - DIA DA SEMANA + COMPANHIA
# ============================================================

for companhia in top_10:

    dados = (
        dia_top[
            dia_top["AIRLINE"] == companhia
        ]
        .sort_values("DAY_OF_WEEK")
    )

    axes[1, 1].plot(
        dados["DAY_OF_WEEK"],
        dados["taxa_ajustada"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=companhia
    )


axes[1, 1].set_title(
    "Adjusted delay rate by day and airline"
)

axes[1, 1].set_xlabel(
    "Day of week"
)

axes[1, 1].set_ylabel(
    "Adjusted delayed flights (%)"
)

maior = dia_top[
    "taxa_ajustada"
].max()

axes[1, 1].set_ylim(
    0,
    maior + 5
)

axes[1, 1].set_xticks(
    range(1, 8)
)

axes[1, 1].grid()

axes[1, 1].legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)


# ============================================================
# MOSTRAR
# ============================================================

plt.show()