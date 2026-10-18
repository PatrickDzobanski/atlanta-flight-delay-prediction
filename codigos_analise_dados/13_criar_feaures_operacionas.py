import pandas as pd
from pathlib import Path


# ============================================================
# CAMINHOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
)

# Dataset ORIGINAL completo
arquivo_entrada = (
    pasta
    / "dataset"
    / "flights.csv"
)

# Dataset reduzido para nossa análise operacional
arquivo_saida = (
    pasta
    / "dataset_filtrado"
    / "flights_2015_atl_operacional.csv"
)


# ============================================================
# COLUNAS NECESSÁRIAS
# ============================================================

colunas = [

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

    # Horários programados
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME",

    # Saída real
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",

    # Característica do voo
    "DISTANCE",

    # Situação
    "CANCELLED",
    "DIVERTED"
]


# ============================================================
# LER DATASET ORIGINAL EM BLOCOS
# ============================================================

partes = []

print("\nLendo dataset original...")


for bloco in pd.read_csv(
    arquivo_entrada,
    usecols=colunas,
    chunksize=300_000
):

    # Somente 2015
    bloco = bloco[
        bloco["YEAR"] == 2015
    ]


    # Manter:
    #
    # ATL -> outra cidade
    # outra cidade -> ATL

    bloco_atl = bloco[
        (bloco["ORIGIN_AIRPORT"] == "ATL")
        |
        (bloco["DESTINATION_AIRPORT"] == "ATL")
    ].copy()


    partes.append(
        bloco_atl
    )


# ============================================================
# JUNTAR OS BLOCOS
# ============================================================

df = pd.concat(
    partes,
    ignore_index=True
)

del partes


# ============================================================
# VERIFICAÇÕES
# ============================================================

saidas_atl = (
    df["ORIGIN_AIRPORT"]
    .eq("ATL")
    .sum()
)

chegadas_atl = (
    df["DESTINATION_AIRPORT"]
    .eq("ATL")
    .sum()
)


print("\n==============================")
print("RESULTADO")
print("==============================")

print(
    "Total de voos:",
    len(df)
)

print(
    "Voos saindo de ATL:",
    saidas_atl
)

print(
    "Voos chegando em ATL:",
    chegadas_atl
)


# ============================================================
# VERIFICAR INFORMAÇÕES DO VOO ANTERIOR
# ============================================================

print("\n==============================")
print("EXEMPLOS")
print("==============================")

print(
    df[
        [
            "AIRLINE",
            "FLIGHT_NUMBER",
            "TAIL_NUMBER",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "SCHEDULED_DEPARTURE",
            "DEPARTURE_TIME",
            "DEPARTURE_DELAY",
            "SCHEDULED_ARRIVAL"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SALVAR
# ============================================================

arquivo_saida.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    arquivo_saida,
    index=False
)


print("\n==============================")
print("ARQUIVO SALVO")
print("==============================")

print(
    arquivo_saida
)