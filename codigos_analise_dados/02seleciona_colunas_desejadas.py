import pandas as pd
from pathlib import Path

# ==============================
# CAMINHOS      
# ==============================

pasta = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset")

arquivo_entrada = pasta / "atlanta_flights_2015.csv"

df = pd.read_csv(arquivo_entrada)

colunas = [
    # Data
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",

    # Identificação do voo
    "AIRLINE",
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",

    # Horários programados
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME",

    # Horários reais
    "DEPARTURE_TIME",
    "ARRIVAL_TIME",

    # Atrasos
    "DEPARTURE_DELAY",
    "ARRIVAL_DELAY",

    # Características do voo
    "DISTANCE",

    # Situação do voo
    "CANCELLED",
    "DIVERTED"
]

df_modelo = df[colunas]

arquivo_saida = pasta / "atlanta_flights_2015_modelo_19col.csv"
df_modelo.to_csv(arquivo_saida, index=False)