import pandas as pd
from pathlib import Path

arquivo_entrada = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset\atlanta_flights_sem_cancelamento.csv")
df = pd.read_csv(arquivo_entrada)

def classificar_periodo(horario):
    if horario < 6:
        return "Madrugada"
    elif horario < 12:
        return "Manhã"
    elif horario < 18:
        return "Tarde"
    else:
        return "Noite"


df["HOUR"] = df["SCHEDULED_DEPARTURE"] // 100
df.loc[df["HOUR"] == 24, "HOUR"] = 0
df["DEPARTURE_MINUTES"] = (df["HOUR"] * 60 + df["SCHEDULED_DEPARTURE"] % 100)
df["PERIODO"] = df["HOUR"].apply(classificar_periodo)


pd.set_option("display.max_columns", None)
print(df.head(10))

arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset\atlanta_flights_com_horario_sem_cancelamento_periodo.csv")
df.to_csv(arquivo_saida, index=False)