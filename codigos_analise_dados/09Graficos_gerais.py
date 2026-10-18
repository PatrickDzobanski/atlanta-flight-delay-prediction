import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PASTA COM OS CSVs PROCESSADOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado"
)


# ============================================================
# CARREGAR OS CSVs
# ============================================================

df_hora = pd.read_csv(pasta / "atraso_por_hora.csv")
df_periodo = pd.read_csv(pasta / "atraso_por_periodo.csv")
df_dia = pd.read_csv(pasta / "atraso_por_dia.csv")
df_mes = pd.read_csv(pasta / "resumo_atraso_mes.csv")


# ============================================================
# PREPARAR DADOS DO MÊS
# ============================================================
# resumo_atraso_mes possui várias companhias dentro de cada mês.
# Somamos os voos e atrasos de todas as companhias.

mes_geral = (
    df_mes.groupby("MONTH")
    .agg(
        total_voos=("total_voos", "sum"),
        total_atrasos=("total_atrasos", "sum")
    )
)

mes_geral["percentual_atrasos"] = (
    mes_geral["total_atrasos"]
    / mes_geral["total_voos"]
    * 100
)


# ============================================================
# COLOCAR PERÍODOS NA ORDEM CORRETA
# ============================================================

ordem_periodos = [
    "Madrugada",
    "Manhã",
    "Tarde",
    "Noite"
]

df_periodo = (
    df_periodo
    .set_index("PERIODO")
    .reindex(ordem_periodos)
    .reset_index()
)


# ============================================================
# CRIAR UMA JANELA COM 4 GRÁFICOS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))


# ------------------------------------------------------------
# GRÁFICO 1 - ATRASO POR HORA
# ------------------------------------------------------------

df_hora["DELAYED"] = df_hora["DELAYED"] * 100

axes[0, 0].plot(
    df_hora["HOUR"],
    df_hora["DELAYED"],
    marker="o"
)

axes[0, 0].set_title("Percentage of delayed flights by hour")
axes[0, 0].set_xlabel("Hour")
axes[0, 0].set_ylabel("Delayed flights (%)")

maior_valor = df_hora["DELAYED"].max()
axes[0, 0].set_ylim(0, maior_valor + 5)

axes[0, 0].grid()
axes[0, 0].set_xticks(df_hora["HOUR"])

# ------------------------------------------------------------
# GRÁFICO 2 - ATRASO POR MÊS
# ------------------------------------------------------------

axes[0, 1].plot(
    mes_geral.index,
    mes_geral["percentual_atrasos"],
    marker="o"
)

axes[0, 1].set_title("Percentage of delayed flights by month")
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Delayed flights (%)")

maior_valor_mes = mes_geral["percentual_atrasos"].max()
axes[0, 1].set_ylim(0, maior_valor_mes + 5)

axes[0, 1].grid()
axes[0, 1].set_xticks(range(1, 13))

# ------------------------------------------------------------
# GRÁFICO 3 - ATRASO POR DIA DA SEMANA
# ------------------------------------------------------------

df_dia["DELAYED"] *= 100

axes[1, 0].bar(
    df_dia["DAY_OF_WEEK"],
    df_dia["DELAYED"]
)

axes[1, 0].set_title("Percentage of delayed flights by day")
axes[1, 0].set_xlabel("Day of week")
axes[1, 0].set_ylabel("Delayed flights (%)")

maior_valor_dia = df_dia["DELAYED"].max()
axes[1, 0].set_ylim(0, maior_valor_dia + 5)

axes[1, 0].grid(axis="y")


# ------------------------------------------------------------
# GRÁFICO 4 - ATRASO POR PERÍODO
# ------------------------------------------------------------

df_periodo["DELAYED"] *= 100

axes[1, 1].bar(
    df_periodo["PERIODO"],
    df_periodo["DELAYED"]
)

axes[1, 1].set_title("Percentage of delayed flights by period")
axes[1, 1].set_xlabel("Period of day")
axes[1, 1].set_ylabel("Delayed flights (%)")

maior_valor_periodo = df_periodo["DELAYED"].max()
axes[1, 1].set_ylim(0, maior_valor_periodo + 5)

axes[1, 1].grid(axis="y")


# ============================================================
# ORGANIZAR E MOSTRAR
# ============================================================

plt.tight_layout()
plt.show()