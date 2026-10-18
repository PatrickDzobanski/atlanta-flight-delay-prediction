import pandas as pd 
from pathlib import Path

arquivo_entrada = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset\atlanta_flights_2015_modelo_19col.csv")
df = pd.read_csv(arquivo_entrada)

# separando os voos que foram cancelados
df_atrasos = df[(df["CANCELLED"] == 0) & (df["DEPARTURE_DELAY"].notna())].copy()
# criando uma nova coluna para indicar se o voo atrasou ou nao
df_atrasos["DELAYED"] = df_atrasos["DEPARTURE_DELAY"] >= 15

arquivo_saida = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset\atlanta_flights_sem_cancelamento.csv")
df_atrasos.to_csv(arquivo_saida, index=False)

print('============= mostra quantos voos atrasaram e quantos nao atrasaram =================')
print(df_atrasos["DELAYED"].value_counts())
print('============= mostra a porcentagem de voos atrasados e nao atrasados =================')
print(df_atrasos["DELAYED"].value_counts(normalize=True) * 100)