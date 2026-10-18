import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURAÇÕES
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
)

arquivo_entrada = (
    pasta
    / "dataset_filtrado"
    / "flights_2015_atl_operacional.csv"
)

arquivo_saida = (
    pasta
    / "dataset_filtrado"
    / "atlanta_flights_features_operacionais.csv"
)


# A previsão será feita 30 minutos antes
# da saída programada do voo de ATL.
HORIZONTE_PREVISAO_MIN = 30


# ============================================================
# CARREGAR DATASET
# ============================================================

print("\nCarregando dataset...")

df = pd.read_csv(
    arquivo_entrada
)

print(
    "Linhas carregadas:",
    len(df)
)


# ============================================================
# GARANTIR COLUNAS NUMÉRICAS
# ============================================================

colunas_numericas = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME",
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "DISTANCE",
    "CANCELLED",
    "DIVERTED"
]

for coluna in colunas_numericas:

    df[coluna] = pd.to_numeric(
        df[coluna],
        errors="coerce"
    )


# ============================================================
# CRIAR DATA
# ============================================================

df["DATA"] = pd.to_datetime(
    dict(
        year=df["YEAR"],
        month=df["MONTH"],
        day=df["DAY"]
    ),
    errors="coerce"
)


# ============================================================
# FUNÇÃO HHMM -> MINUTOS
# ============================================================

def horario_para_minutos(serie):

    serie = pd.to_numeric(
        serie,
        errors="coerce"
    )

    hora = (
        serie // 100
    )

    minuto = (
        serie % 100
    )

    # 2400 será tratado separadamente
    hora = hora % 24

    return (
        hora * 60
        + minuto
    )


# ============================================================
# HORÁRIO PROGRAMADO DE PARTIDA
# ============================================================

df["MIN_SCHED_DEP"] = (
    horario_para_minutos(
        df["SCHEDULED_DEPARTURE"]
    )
)


# Caso raro de 2400
dia_extra_saida = (
    df["SCHEDULED_DEPARTURE"]
    == 2400
)


df["DT_SCHED_DEP"] = (
    df["DATA"]
    +
    pd.to_timedelta(
        df["MIN_SCHED_DEP"],
        unit="m"
    )
    +
    pd.to_timedelta(
        dia_extra_saida.astype(int),
        unit="D"
    )
)


# ============================================================
# VOOS DE ATL QUE QUEREMOS PREVER
# ============================================================

df_atl = df[
    (df["ORIGIN_AIRPORT"] == "ATL")
    &
    (df["CANCELLED"] == 0)
    &
    (df["DEPARTURE_DELAY"].notna())
    &
    (df["DT_SCHED_DEP"].notna())
].copy()


# ============================================================
# TARGET
# ============================================================

df_atl["DELAYED"] = (
    df_atl["DEPARTURE_DELAY"]
    >= 15
).astype(int)


# ============================================================
# MOMENTO DA PREVISÃO
# ============================================================

df_atl["DT_PREVISAO"] = (
    df_atl["DT_SCHED_DEP"]
    -
    pd.Timedelta(
        minutes=HORIZONTE_PREVISAO_MIN
    )
)


# ============================================================
# FEATURES DE HORÁRIO
# ============================================================

df_atl["HOUR"] = (
    df_atl["MIN_SCHED_DEP"]
    // 60
).astype(int)


df_atl["DEPARTURE_MINUTES"] = (
    df_atl["MIN_SCHED_DEP"]
)


def classificar_periodo(hora):

    if hora < 6:
        return "Madrugada"

    elif hora < 12:
        return "Manhã"

    elif hora < 18:
        return "Tarde"

    else:
        return "Noite"


df_atl["PERIODO"] = (
    df_atl["HOUR"]
    .apply(classificar_periodo)
)


# ============================================================
# 1. CONGESTIONAMENTO PROGRAMADO EM ATL
# ============================================================
#
# Quantos outros voos estão programados
# para sair de ATL em uma janela de:
#
# -30 min até +30 min
#
# Como são horários PROGRAMADOS,
# essa informação já é conhecida antes.
#
# ============================================================

saidas_atl = df[
    (df["ORIGIN_AIRPORT"] == "ATL")
    &
    (df["DT_SCHED_DEP"].notna())
].copy()


tempos_saidas = np.sort(
    saidas_atl[
        "DT_SCHED_DEP"
    ]
    .values
    .astype(
        "datetime64[ns]"
    )
)


tempos_voos = (
    df_atl[
        "DT_SCHED_DEP"
    ]
    .values
    .astype(
        "datetime64[ns]"
    )
)


inicio = np.searchsorted(
    tempos_saidas,
    tempos_voos
    -
    np.timedelta64(
        30,
        "m"
    ),
    side="left"
)


fim = np.searchsorted(
    tempos_saidas,
    tempos_voos
    +
    np.timedelta64(
        30,
        "m"
    ),
    side="right"
)


df_atl[
    "CONGESTIONAMENTO_ATL"
] = (
    fim
    - inicio
    - 1
)


# ============================================================
# 2. ESTADO RECENTE DA COMPANHIA
# ============================================================
#
# Vamos olhar os voos da mesma companhia
# que REALMENTE já partiram de ATL
# nas 2 horas anteriores ao momento da previsão.
#
# ============================================================

eventos = df[
    (df["ORIGIN_AIRPORT"] == "ATL")
    &
    (df["CANCELLED"] == 0)
    &
    (df["DEPARTURE_DELAY"].notna())
    &
    (df["DT_SCHED_DEP"].notna())
].copy()


# Saída real reconstruída:
#
# horário programado + atraso
#
eventos["DT_ACT_DEP"] = (
    eventos["DT_SCHED_DEP"]
    +
    pd.to_timedelta(
        eventos["DEPARTURE_DELAY"],
        unit="m"
    )
)


eventos["ATRASADO"] = (
    eventos["DEPARTURE_DELAY"]
    >= 15
).astype(int)


df_atl[
    "QTD_VOOS_COMPANHIA_2H"
] = 0


df_atl[
    "QTD_ATRASOS_COMPANHIA_2H"
] = 0


df_atl[
    "TAXA_ATRASO_COMPANHIA_2H"
] = np.nan


# ============================================================
# CALCULAR POR COMPANHIA
# ============================================================

for companhia in (
    df_atl["AIRLINE"]
    .dropna()
    .unique()
):

    eventos_companhia = (
        eventos[
            eventos["AIRLINE"]
            == companhia
        ]
        .sort_values(
            "DT_ACT_DEP"
        )
    )


    mascara = (
        df_atl["AIRLINE"]
        == companhia
    )


    alvos = (
        df_atl.loc[
            mascara
        ]
    )


    if eventos_companhia.empty:
        continue


    tempos_eventos = (
        eventos_companhia[
            "DT_ACT_DEP"
        ]
        .values
        .astype(
            "datetime64[ns]"
        )
    )


    atrasos_eventos = (
        eventos_companhia[
            "ATRASADO"
        ]
        .to_numpy()
    )


    # Soma acumulada
    soma_atrasos = np.concatenate(
        [
            [0],
            np.cumsum(
                atrasos_eventos
            )
        ]
    )


    tempos_previsao = (
        alvos[
            "DT_PREVISAO"
        ]
        .values
        .astype(
            "datetime64[ns]"
        )
    )


    inicio_janela = (
        tempos_previsao
        -
        np.timedelta64(
            2,
            "h"
        )
    )


    pos_inicio = np.searchsorted(
        tempos_eventos,
        inicio_janela,
        side="left"
    )


    pos_fim = np.searchsorted(
        tempos_eventos,
        tempos_previsao,
        side="left"
    )


    qtd_voos = (
        pos_fim
        - pos_inicio
    )


    qtd_atrasos = (
        soma_atrasos[pos_fim]
        -
        soma_atrasos[pos_inicio]
    )


    taxa = np.divide(
        qtd_atrasos,
        qtd_voos,

        out=np.full(
            len(qtd_voos),
            np.nan,
            dtype=float
        ),

        where=(
            qtd_voos > 0
        )
    )


    df_atl.loc[
        mascara,
        "QTD_VOOS_COMPANHIA_2H"
    ] = qtd_voos


    df_atl.loc[
        mascara,
        "QTD_ATRASOS_COMPANHIA_2H"
    ] = qtd_atrasos


    df_atl.loc[
        mascara,
        "TAXA_ATRASO_COMPANHIA_2H"
    ] = taxa


# ============================================================
# 3. VOO ANTERIOR DA MESMA AERONAVE
# ============================================================
#
# Agora procuramos:
#
# OUTRO AEROPORTO -> ATL
#
# com o mesmo TAIL_NUMBER.
#
# NÃO vamos usar:
#
# ARRIVAL_TIME
# ARRIVAL_DELAY
#
# ============================================================

chegadas = df[
    (df["DESTINATION_AIRPORT"] == "ATL")
    &
    (df["TAIL_NUMBER"].notna())
    &
    (df["SCHEDULED_DEPARTURE"].notna())
    &
    (df["SCHEDULED_ARRIVAL"].notna())
    &
    (df["SCHEDULED_TIME"].notna())
].copy()


# ============================================================
# HORÁRIO PROGRAMADO DE CHEGADA EM ATL
# ============================================================

chegadas["MIN_SCHED_ARR"] = (
    horario_para_minutos(
        chegadas[
            "SCHEDULED_ARRIVAL"
        ]
    )
)


# ============================================================
# DESCOBRIR SE CHEGADA É NO MESMO DIA OU DIA SEGUINTE
# ============================================================
#
# Aqui usamos SCHEDULED_TIME para ajudar.
#
# Para cada voo testamos:
#
# chegada no mesmo dia
# chegada no dia seguinte
# chegada dois dias depois
#
# e escolhemos a opção temporal mais coerente.
#
# Isso ajuda por causa dos fusos horários.
#
# ============================================================

dep_min = (
    chegadas[
        "MIN_SCHED_DEP"
    ]
    .to_numpy(
        dtype=float
    )
)


arr_min = (
    chegadas[
        "MIN_SCHED_ARR"
    ]
    .to_numpy(
        dtype=float
    )
)


tempo_voo = (
    chegadas[
        "SCHEDULED_TIME"
    ]
    .to_numpy(
        dtype=float
    )
)


dep_dia_extra = (
    chegadas[
        "SCHEDULED_DEPARTURE"
    ]
    .eq(2400)
    .astype(int)
    .to_numpy()
)


dep_total = (
    dep_min
    +
    1440
    * dep_dia_extra
)


# Testar chegada no dia 0, 1 ou 2
candidatos = np.column_stack(
    [
        arr_min,
        arr_min + 1440,
        arr_min + 2880
    ]
)


# Diferença:
#
# chegada - partida - duração
#
# O resultado representa aproximadamente
# a diferença de fuso horário.
#
erro = np.abs(
    candidatos
    -
    dep_total[:, None]
    -
    tempo_voo[:, None]
)


melhor_dia = np.argmin(
    erro,
    axis=1
)


chegadas[
    "DT_SCHED_ARR_ATL"
] = (
    chegadas["DATA"]
    +
    pd.to_timedelta(
        chegadas[
            "MIN_SCHED_ARR"
        ],
        unit="m"
    )
    +
    pd.to_timedelta(
        melhor_dia,
        unit="D"
    )
)


# ============================================================
# CONVERTER A PARTIDA DO VOO ANTERIOR
# PARA A LINHA DO TEMPO DE ATL
# ============================================================
#
# Exemplo:
#
# chegada prevista ATL = 07:30
# duração prevista      = 120 min
#
# saída prevista,
# representada no relógio de ATL:
#
# 07:30 - 120 min = 05:30
#
# ============================================================

chegadas[
    "DT_SCHED_DEP_ANTERIOR_ATL"
] = (
    chegadas[
        "DT_SCHED_ARR_ATL"
    ]
    -
    pd.to_timedelta(
        chegadas[
            "SCHEDULED_TIME"
        ],
        unit="m"
    )
)


# ============================================================
# SAÍDA REAL DO VOO ANTERIOR
# ============================================================
#
# Só usamos o atraso para reconstruir
# quando aquela aeronave realmente saiu.
#
# ============================================================

chegadas[
    "DT_ACT_DEP_ANTERIOR_ATL"
] = (
    chegadas[
        "DT_SCHED_DEP_ANTERIOR_ATL"
    ]
    +
    pd.to_timedelta(
        chegadas[
            "DEPARTURE_DELAY"
        ],
        unit="m"
    )
)


# ============================================================
# PREPARAR COLUNAS DO VOO ANTERIOR
# ============================================================

chegadas = chegadas[
    [
        "TAIL_NUMBER",
        "DT_SCHED_ARR_ATL",
        "DT_SCHED_DEP_ANTERIOR_ATL",
        "DT_ACT_DEP_ANTERIOR_ATL",
        "DEPARTURE_DELAY",
        "ORIGIN_AIRPORT",
        "CANCELLED",
        "DIVERTED"
    ]
].copy()


chegadas = chegadas.rename(
    columns={
        "DEPARTURE_DELAY":
            "ATRASO_SAIDA_ANTERIOR_FINAL",

        "ORIGIN_AIRPORT":
            "ORIGEM_VOO_ANTERIOR",

        "CANCELLED":
            "CANCELLED_ANTERIOR",

        "DIVERTED":
            "DIVERTED_ANTERIOR"
    }
)


# ============================================================
# IDENTIFICADOR DOS VOOS ATL
# ============================================================

df_atl["ROW_ID"] = np.arange(
    len(df_atl)
)


# ============================================================
# PREPARAR PARA MERGE
# ============================================================

alvos_tail = (
    df_atl[
        df_atl[
            "TAIL_NUMBER"
        ].notna()
    ][
        [
            "ROW_ID",
            "TAIL_NUMBER",
            "DT_SCHED_DEP"
        ]
    ]
    .sort_values(
        [
            "DT_SCHED_DEP",
            "TAIL_NUMBER"
        ]
    )
)


chegadas = (
    chegadas
    .sort_values(
        [
            "DT_SCHED_ARR_ATL",
            "TAIL_NUMBER"
        ]
    )
)


# ============================================================
# ENCONTRAR ÚLTIMO VOO DA AERONAVE PARA ATL
# ============================================================

encontrados = pd.merge_asof(
    alvos_tail,
    chegadas,

    left_on="DT_SCHED_DEP",
    right_on="DT_SCHED_ARR_ATL",

    by="TAIL_NUMBER",

    direction="backward",

    allow_exact_matches=True
)


encontrados = (
    encontrados
    .set_index(
        "ROW_ID"
    )
)


# ============================================================
# ADICIONAR RESULTADOS AO DATAFRAME
# ============================================================

colunas_anterior = [
    "DT_SCHED_ARR_ATL",
    "DT_SCHED_DEP_ANTERIOR_ATL",
    "DT_ACT_DEP_ANTERIOR_ATL",
    "ATRASO_SAIDA_ANTERIOR_FINAL",
    "ORIGEM_VOO_ANTERIOR"
]


for coluna in colunas_anterior:

    df_atl[coluna] = (
        df_atl[
            "ROW_ID"
        ]
        .map(
            encontrados[coluna]
        )
    )


# ============================================================
# TEM VOO ANTERIOR?
# ============================================================
#
# Limitamos a 36 horas.
#
# Isso evita associar um voo de vários dias atrás
# à aeronave atual.
#
# ============================================================

horas_desde_chegada_programada = (
    (
        df_atl[
            "DT_SCHED_DEP"
        ]
        -
        df_atl[
            "DT_SCHED_ARR_ATL"
        ]
    )
    .dt.total_seconds()
    /
    3600
)


df_atl[
    "TEM_VOO_ANTERIOR"
] = (
    horas_desde_chegada_programada
    .between(
        0,
        36,
        inclusive="both"
    )
).astype(int)


# ============================================================
# VOO ANTERIOR JÁ HAVIA SAÍDO?
# ============================================================
#
# IMPORTANTE:
#
# Só podemos usar DEPARTURE_DELAY do voo anterior
# se ele já tiver decolado ANTES do momento da previsão.
#
# ============================================================

ja_saiu = (
    (df_atl["TEM_VOO_ANTERIOR"] == 1)
    &
    (
        df_atl[
            "DT_ACT_DEP_ANTERIOR_ATL"
        ].notna()
    )
    &
    (
        df_atl[
            "DT_ACT_DEP_ANTERIOR_ATL"
        ]
        <=
        df_atl[
            "DT_PREVISAO"
        ]
    )
)


# ============================================================
# FEATURE:
# VOO ANTERIOR AINDA NÃO SAIU
# ============================================================

df_atl[
    "VOO_ANTERIOR_AINDA_NAO_SAIU"
] = np.nan


tem_anterior = (
    df_atl[
        "TEM_VOO_ANTERIOR"
    ]
    == 1
)


df_atl.loc[
    tem_anterior,
    "VOO_ANTERIOR_AINDA_NAO_SAIU"
] = (
    ~ja_saiu[
        tem_anterior
    ]
).astype(int)


# ============================================================
# FEATURE:
# ATRASO DA SAÍDA DO VOO ANTERIOR
# ============================================================
#
# Exemplo:
#
# previsto = 06:00
# real     = 06:35
#
# ATRASO = 35 minutos
#
# Se o voo ainda não tinha saído no momento
# da previsão:
#
# NaN
#
# ============================================================

df_atl[
    "ATRASO_SAIDA_VOO_ANTERIOR"
] = np.nan


df_atl.loc[
    ja_saiu,
    "ATRASO_SAIDA_VOO_ANTERIOR"
] = (
    df_atl.loc[
        ja_saiu,
        "ATRASO_SAIDA_ANTERIOR_FINAL"
    ]
)


# ============================================================
# REMOVER ASSOCIAÇÕES MUITO ANTIGAS
# ============================================================

mascara_sem_anterior = (
    df_atl[
        "TEM_VOO_ANTERIOR"
    ]
    == 0
)


df_atl.loc[
    mascara_sem_anterior,
    [
        "ATRASO_SAIDA_VOO_ANTERIOR",
        "VOO_ANTERIOR_AINDA_NAO_SAIU"
    ]
] = np.nan


# ============================================================
# MOSTRAR EXEMPLOS
# ============================================================

colunas_teste = [
    "AIRLINE",
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
    "DESTINATION_AIRPORT",
    "DT_SCHED_DEP",
    "DT_PREVISAO",

    "CONGESTIONAMENTO_ATL",

    "QTD_VOOS_COMPANHIA_2H",
    "QTD_ATRASOS_COMPANHIA_2H",
    "TAXA_ATRASO_COMPANHIA_2H",

    "TEM_VOO_ANTERIOR",
    "ORIGEM_VOO_ANTERIOR",
    "ATRASO_SAIDA_VOO_ANTERIOR",
    "VOO_ANTERIOR_AINDA_NAO_SAIU",

    "DELAYED"
]


print("\n")
print("=" * 110)
print("FEATURES OPERACIONAIS")
print("=" * 110)


print(
    df_atl[
        colunas_teste
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# RESUMO
# ============================================================

print("\n")
print("=" * 70)
print("RESUMO")
print("=" * 70)


print(
    "Voos ATL para previsão:",
    len(df_atl)
)


print(
    "Taxa de atraso:",
    f"{df_atl['DELAYED'].mean() * 100:.2f}%"
)


print(
    "Com voo anterior identificado:",
    int(
        df_atl[
            "TEM_VOO_ANTERIOR"
        ]
        .sum()
    )
)


print(
    "Percentual com voo anterior:",
    f"{df_atl['TEM_VOO_ANTERIOR'].mean() * 100:.2f}%"
)


print(
    "Voo anterior ainda não tinha saído:",
    int(
        (
            df_atl[
                "VOO_ANTERIOR_AINDA_NAO_SAIU"
            ]
            == 1
        )
        .sum()
    )
)


print(
    "Com atraso do voo anterior disponível:",
    int(
        df_atl[
            "ATRASO_SAIDA_VOO_ANTERIOR"
        ]
        .notna()
        .sum()
    )
)


# ============================================================
# COLUNAS DO DATASET FINAL
# ============================================================

colunas_saida = [

    # Data
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",

    # Identificação
    "AIRLINE",
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",

    # Informações programadas
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME",
    "DISTANCE",

    # Features de horário
    "HOUR",
    "DEPARTURE_MINUTES",
    "PERIODO",

    # Features operacionais ATL
    "CONGESTIONAMENTO_ATL",

    "QTD_VOOS_COMPANHIA_2H",
    "QTD_ATRASOS_COMPANHIA_2H",
    "TAXA_ATRASO_COMPANHIA_2H",

    # Features da aeronave
    "TEM_VOO_ANTERIOR",
    "ATRASO_SAIDA_VOO_ANTERIOR",
    "VOO_ANTERIOR_AINDA_NAO_SAIU",

    # Target
    "DELAYED"
]


df_saida = (
    df_atl[
        colunas_saida
    ]
    .copy()
)


# ============================================================
# ORDENAR
# ============================================================

df_saida = (
    df_saida
    .sort_values(
        [
            "YEAR",
            "MONTH",
            "DAY",
            "SCHEDULED_DEPARTURE"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SALVAR
# ============================================================

arquivo_saida.parent.mkdir(
    parents=True,
    exist_ok=True
)


df_saida.to_csv(
    arquivo_saida,
    index=False
)


print("\n")
print("=" * 70)
print("ARQUIVO SALVO")
print("=" * 70)

print(
    arquivo_saida
)