import pandas as pd
from pathlib import Path        

arquivo_entrada = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset\atlanta_flights_com_horario_sem_cancelamento_periodo.csv")
df = pd.read_csv(arquivo_entrada)

novo_df = df.copy()

# por companhia aerea
# 1. Percentual de atraso por hora e companhia
atraso_hora_companhia = (
    df.groupby(["HOUR", "AIRLINE"])
    .agg(
        percentual_atrasos=("DELAYED", "mean"),
        num_voos=("DELAYED", "size")
    )
)

# 2. Percentual de atraso por período e companhia
atraso_periodo_companhia = (
    df.groupby(["PERIODO", "AIRLINE"])
    .agg(
        percentual_atrasos=("DELAYED", "mean"),
        num_voos=("DELAYED", "size")
    ) 
)

# 3. Percentual de atraso por dia da semana e companhia
atraso_dia_companhia = (
    df.groupby(["DAY_OF_WEEK", "AIRLINE"])
    .agg(
        percentual_atrasos=("DELAYED", "mean"),
        num_voos=("DELAYED", "size")
    )
)
#==============================
# geral 
# 1. Percentual de atraso por hora
atraso_por_hora = (
    df.groupby("HOUR")["DELAYED"].mean() 
)

# 2. Percentual de atraso por período do dia
atraso_por_periodo = (
    df.groupby("PERIODO")["DELAYED"].mean()
)

# 3. Percentual de atraso por dia da semana
atraso_por_dia = (
    df.groupby("DAY_OF_WEEK")["DELAYED"].mean() 
)


arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\atraso_hora_companhia.csv")
atraso_hora_companhia.reset_index().to_csv(arquivo_saida, index=False)
arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\atraso_periodo_companhia.csv")
atraso_periodo_companhia.reset_index().to_csv(arquivo_saida, index=False)  
arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\atraso_dia_companhia.csv")
atraso_dia_companhia.reset_index().to_csv(arquivo_saida, index=False)
arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\atraso_por_hora.csv")
atraso_por_hora.reset_index().to_csv(arquivo_saida, index=False)
arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\atraso_por_periodo.csv")
atraso_por_periodo.reset_index().to_csv(arquivo_saida, index=False)
arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\atraso_por_dia.csv")
atraso_por_dia.reset_index().to_csv(arquivo_saida, index=False)