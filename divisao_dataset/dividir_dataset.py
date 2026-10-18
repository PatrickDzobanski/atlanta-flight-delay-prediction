import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# CAMINHOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
)

arquivo_entrada = (
    pasta
    / "dataset_filtrado"
    / "atlanta_flights_features_operacionais.csv"
)

arquivo_treino = (
    pasta
    / "dataset_filtrado"
    / "arquivo_treino_operacional.csv"
)

arquivo_calibracao = (
    pasta
    / "dataset_filtrado"
    / "arquivo_calibracao_operacional.csv"
)

arquivo_teste = (
    pasta
    / "dataset_filtrado"
    / "arquivo_teste_operacional.csv"
)


# ============================================================
# CARREGAR DATASET
# ============================================================

print("\nCarregando dataset...")

df = pd.read_csv(
    arquivo_entrada
)

print(
    "Total de voos:",
    len(df)
)


# ============================================================
# GARANTIR TARGET COMO 0/1
# ============================================================

df["DELAYED"] = (
    df["DELAYED"]
    .astype(int)
)


# ============================================================
# CRIAR GRUPOS PARA ESTRATIFICAÇÃO
# ============================================================
#
# Queremos que os três datasets mantenham proporções
# semelhantes de:
#
# - mês
# - período do dia
# - atraso / não atraso
#
# Em alguns datasets reais, a combinação exata (mês + período + atraso)
# pode ter classes muito raras (apenas 1 amostra). Nesse caso, o
# sklearn rejeita a estratificação porque a classe fica abaixo do
# mínimo exigido. A solução é tentar a estratificação mais detalhada
# e fazer fallback para níveis mais gerais até encontrar uma opção
# válida.
#
# Exemplo:
#
# 1_Manhã_0
# 1_Manhã_1
# 1_Tarde_0
# 1_Tarde_1
# ...
#
# ============================================================

df["GRUPO_ESTRATIFICACAO"] = (
    df["MONTH"].astype(str)
    + "_"
    + df["PERIODO"].astype(str)
    + "_"
    + df["DELAYED"].astype(str)
)

df["GRUPO_ESTRATIFICACAO_MES_DELAYED"] = (
    df["MONTH"].astype(str)
    + "_"
    + df["DELAYED"].astype(str)
)

df["GRUPO_ESTRATIFICACAO_DELAYED"] = df["DELAYED"].astype(str)


def obter_stratificacao_ou_none(df, colunas):
    for coluna in colunas:
        contagem = df[coluna].value_counts()
        if contagem.min() >= 2:
            return coluna
    return None


coluna_estratificacao = obter_stratificacao_ou_none(
    df,
    [
        "GRUPO_ESTRATIFICACAO",
        "GRUPO_ESTRATIFICACAO_MES_DELAYED",
        "GRUPO_ESTRATIFICACAO_DELAYED",
    ]
)

if coluna_estratificacao is not None:
    print(f"Usando estratificação por: {coluna_estratificacao}")
else:
    print("Nenhuma estratificação válida encontrada; dividindo sem stratify.")


# ============================================================
# PRIMEIRA DIVISÃO
# ============================================================
#
# 70% treino
# 30% temporário
#
# ============================================================

parametros_primeira_divisao = {
    "test_size": 0.30,
    "random_state": 42,
}

if coluna_estratificacao is not None:
    parametros_primeira_divisao["stratify"] = df[coluna_estratificacao]


df_treino, df_temp = train_test_split(
    df,
    **parametros_primeira_divisao
)


# ============================================================
# SEGUNDA DIVISÃO
# ============================================================
#
# Dos 30% restantes:
#
# metade -> calibração = 15%
# metade -> teste      = 15%
#
# ============================================================

coluna_estratificacao_temp = obter_stratificacao_ou_none(
    df_temp,
    [
        "GRUPO_ESTRATIFICACAO",
        "GRUPO_ESTRATIFICACAO_MES_DELAYED",
        "GRUPO_ESTRATIFICACAO_DELAYED",
    ]
)

if coluna_estratificacao_temp is not None:
    print(f"Usando estratificação para calibração/teste por: {coluna_estratificacao_temp}")
else:
    print("Nenhuma estratificação válida encontrada para calibração/teste; dividindo sem stratify.")

parametros_segunda_divisao = {
    "test_size": 0.50,
    "random_state": 42,
}

if coluna_estratificacao_temp is not None:
    parametros_segunda_divisao["stratify"] = df_temp[coluna_estratificacao_temp]


df_calibracao, df_teste = train_test_split(
    df_temp,
    **parametros_segunda_divisao
)


# ============================================================
# REMOVER COLUNA AUXILIAR
# ============================================================

colunas_auxiliares = [
    "GRUPO_ESTRATIFICACAO",
    "GRUPO_ESTRATIFICACAO_MES_DELAYED",
    "GRUPO_ESTRATIFICACAO_DELAYED",
]


df_treino = (
    df_treino
    .drop(
        columns=colunas_auxiliares
    )
    .reset_index(drop=True)
)


df_calibracao = (
    df_calibracao
    .drop(
        columns=colunas_auxiliares
    )
    .reset_index(drop=True)
)


df_teste = (
    df_teste
    .drop(
        columns=colunas_auxiliares
    )
    .reset_index(drop=True)
)


# ============================================================
# MOSTRAR TAMANHOS
# ============================================================

print("\n")
print("=" * 60)
print("DIVISÃO DO DATASET")
print("=" * 60)

print(
    "Treino:",
    len(df_treino),
    f"({len(df_treino) / len(df) * 100:.2f}%)"
)

print(
    "Calibração:",
    len(df_calibracao),
    f"({len(df_calibracao) / len(df) * 100:.2f}%)"
)

print(
    "Teste:",
    len(df_teste),
    f"({len(df_teste) / len(df) * 100:.2f}%)"
)


# ============================================================
# TAXA DE ATRASO
# ============================================================

print("\n")
print("=" * 60)
print("TAXA DE ATRASO")
print("=" * 60)

print(
    "Dataset completo:",
    f"{df['DELAYED'].mean() * 100:.2f}%"
)

print(
    "Treino:",
    f"{df_treino['DELAYED'].mean() * 100:.2f}%"
)

print(
    "Calibração:",
    f"{df_calibracao['DELAYED'].mean() * 100:.2f}%"
)

print(
    "Teste:",
    f"{df_teste['DELAYED'].mean() * 100:.2f}%"
)


# ============================================================
# VERIFICAR DISTRIBUIÇÃO POR MÊS
# ============================================================

print("\n")
print("=" * 60)
print("DISTRIBUIÇÃO POR MÊS (%)")
print("=" * 60)


distribuicao_mes = pd.DataFrame({
    "COMPLETO":
        df["MONTH"]
        .value_counts(normalize=True)
        .sort_index()
        * 100,

    "TREINO":
        df_treino["MONTH"]
        .value_counts(normalize=True)
        .sort_index()
        * 100,

    "CALIBRACAO":
        df_calibracao["MONTH"]
        .value_counts(normalize=True)
        .sort_index()
        * 100,

    "TESTE":
        df_teste["MONTH"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
})


print(
    distribuicao_mes.round(2)
)


# ============================================================
# VERIFICAR DISTRIBUIÇÃO POR PERÍODO
# ============================================================

print("\n")
print("=" * 60)
print("DISTRIBUIÇÃO POR PERÍODO (%)")
print("=" * 60)


distribuicao_periodo = pd.DataFrame({
    "COMPLETO":
        df["PERIODO"]
        .value_counts(normalize=True)
        * 100,

    "TREINO":
        df_treino["PERIODO"]
        .value_counts(normalize=True)
        * 100,

    "CALIBRACAO":
        df_calibracao["PERIODO"]
        .value_counts(normalize=True)
        * 100,

    "TESTE":
        df_teste["PERIODO"]
        .value_counts(normalize=True)
        * 100
})


print(
    distribuicao_periodo.round(2)
)


# ============================================================
# SALVAR
# ============================================================

df_treino.to_csv(
    arquivo_treino,
    index=False
)

df_calibracao.to_csv(
    arquivo_calibracao,
    index=False
)

df_teste.to_csv(
    arquivo_teste,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("ARQUIVOS SALVOS")
print("=" * 60)

print(
    "\nTreino:"
)
print(
    arquivo_treino
)

print(
    "\nCalibração:"
)
print(
    arquivo_calibracao
)

print(
    "\nTeste:"
)
print(
    arquivo_teste
)