import pandas as pd 
from pathlib import Path

# ==============================
# CAMINHOS
# ==============================
pasta  = Path(r"C:\Users\patri\Desktop\projeto voos em atraso\dataset")

arquivo_entrada = pasta / "flights.csv"
arquivo_saida = pasta / "atlanta_flights_2015.csv"

# =============================
# configuracoes 
# ==============================

chunck_size =  250000 # tamanho dos blocos de dados que vamos usar 

lista_dfs = [] # lista para armazenar os blocos de dados filtrados

# ==============================
# FILTRAGEM 
# ==============================

for chunk in pd.read_csv(arquivo_entrada, chunksize=chunck_size):
    # filtrar voos saindo de ATL em 2015
    df_ATL_2015 = chunk[(chunk["ORIGIN_AIRPORT"] == "ATL") & (chunk["YEAR"] == 2015)]

    # adicionar o bloco filtrado na lista
    lista_dfs.append(df_ATL_2015)

#==============================
# juntar todos os blocos filtrados em um único DataFrame
#==============================

df_final = pd.concat(lista_dfs, ignore_index=True)

#==============================
# salvar o DataFrame final em um arquivo CSV
#==============================

df_final.to_csv(arquivo_saida, index=False)

print("\n==============================")
print(f"Total de voos saindo de ATL: {len(df_ATL_2015)}")