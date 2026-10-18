import pandas as pd
from pathlib import Path

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
)

arquivo = (
    pasta
    / "dataset_filtrado"
    / "atlanta_flights_features_operacionais.csv"
)

df = pd.read_csv(arquivo)


print("\n==============================")
print("SEM VOO ANTERIOR")
print("==============================")

grupo = df[
    df["TEM_VOO_ANTERIOR"] == 0
]

print(
    "Voos:",
    len(grupo)
)

print(
    "Taxa atraso:",
    f"{grupo['DELAYED'].mean() * 100:.2f}%"
)


print("\n==============================")
print("VOO ANTERIOR JÁ SAIU")
print("==============================")

grupo = df[
    (df["TEM_VOO_ANTERIOR"] == 1)
    &
    (df["VOO_ANTERIOR_AINDA_NAO_SAIU"] == 0)
]

print(
    "Voos:",
    len(grupo)
)

print(
    "Taxa atraso:",
    f"{grupo['DELAYED'].mean() * 100:.2f}%"
)


print("\n==============================")
print("VOO ANTERIOR AINDA NÃO SAIU")
print("==============================")

grupo = df[
    df["VOO_ANTERIOR_AINDA_NAO_SAIU"] == 1
]

print(
    "Voos:",
    len(grupo)
)

print(
    "Taxa atraso:",
    f"{grupo['DELAYED'].mean() * 100:.2f}%"
)


print("\n==============================")
print("ATRASO DO VOO ANTERIOR")
print("==============================")

df_anterior = df[
    df["ATRASO_SAIDA_VOO_ANTERIOR"].notna()
].copy()

df_anterior["FAIXA_ATRASO_ANTERIOR"] = pd.cut(
    df_anterior["ATRASO_SAIDA_VOO_ANTERIOR"],
    bins=[
        -999,
        0,
        15,
        30,
        60,
        9999
    ],
    labels=[
        "<= 0 min",
        "0-15 min",
        "15-30 min",
        "30-60 min",
        "> 60 min"
    ]
)

resumo = (
    df_anterior
    .groupby(
        "FAIXA_ATRASO_ANTERIOR",
        observed=False
    )
    ["DELAYED"]
    .agg(
        quantidade="count",
        taxa_atraso="mean"
    )
)

resumo["taxa_atraso"] *= 100

print(resumo)