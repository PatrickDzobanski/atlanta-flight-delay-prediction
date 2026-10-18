import pandas as pd 
from pathlib import Path

arquivo_entrada = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset\atlanta_flights_sem_cancelamento.csv")
df = pd.read_csv(arquivo_entrada)

atraso_por_destino = (
    df.groupby("DESTINATION_AIRPORT")
    .agg(
        numero_voos=("DELAYED", "size"),
        numero_voos_atrasados=("DELAYED", "sum"),
        percentual_voos_atrasados=("DELAYED", "mean")
    )
)

arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset_filtrado\atraso_por_destino.csv")
atraso_por_destino.reset_index().to_csv(arquivo_saida, index=False)
# print(atraso_por_destino.head(10))
# print(atraso_por_destino.shape[0])