import pandas as pd 
from pathlib import Path

arquivo_entrada = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset\atlanta_flights_sem_cancelamento.csv")
df = pd.read_csv(arquivo_entrada)

atraso_por_mes = (
    df.groupby("MONTH")["DELAYED"]
    .mean()*100
)

atraso_por_mes_companhia= (
    df[df["DELAYED"] == True]
    .groupby(["MONTH", "AIRLINE"])
    .size()
)    

resumo_atraso = (
    df.groupby("AIRLINE")
    .agg(
        total_voos=("DELAYED", "size"),
        total_atrasos=("DELAYED", "sum"),
        percentual_atrasos=("DELAYED", "mean")
    )
)

resumo_atraso_mes = (
    df.groupby(["MONTH", "AIRLINE"])
    .agg(
        total_voos=("DELAYED", "size"),
        total_atrasos=("DELAYED", "sum"),
        percentual_atrasos=("DELAYED", "mean")
    )
)

resumo_atraso_dias_semanas = (
    df.groupby(["DAY_OF_WEEK", "AIRLINE"])
    .agg(
        total_voos=("DELAYED", "size"),
        total_atrasos=("DELAYED", "sum"),
        percentual_atrasos=("DELAYED", "mean")
    )
)

resumo_atraso_mes_dias_semanas = (
    df.groupby(["MONTH", "DAY_OF_WEEK", "AIRLINE"])
    .agg(
        total_voos=("DELAYED", "size"),
        total_atrasos=("DELAYED", "sum"),
        percentual_atrasos=("DELAYED", "mean")
    )
)

resumo_estatistica = (
    df[df["DELAYED"] == True]
    .groupby("AIRLINE")
    .agg(
        atraso_medio=("DEPARTURE_DELAY", "mean"),
        atraso_maximo=("DEPARTURE_DELAY", "max"),
        atraso_mediano=("DEPARTURE_DELAY", "median"),
        atraso_desvio_padrao=("DEPARTURE_DELAY", "std")
        )
)

resumo_completo = resumo_atraso.join(resumo_estatistica, how="left")

# print(resumo_atraso)
# print(resumo_atraso_mes)
# print(resumo_atraso_dias_semanas)
# print(resumo_atraso_mes_dias_semanas)
# print(resumo_completo)

arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\resumo_atraso.csv")
resumo_completo.reset_index().to_csv(arquivo_saida, index=False)
arquivo_saida_mes = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\resumo_atraso_mes.csv")
resumo_atraso_mes.reset_index().to_csv(arquivo_saida_mes, index=False)
arquivo_saida_dias_semanas = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\resumo_atraso_dias_semanas.csv")
resumo_atraso_dias_semanas.reset_index().to_csv(arquivo_saida_dias_semanas, index=False)
