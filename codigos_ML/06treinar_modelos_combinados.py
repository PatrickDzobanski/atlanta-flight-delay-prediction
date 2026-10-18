import pandas as pd
import json

from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


# ============================================================
# CAMINHOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
)

arquivo_treino = (
    pasta
    / "dataset_filtrado"
    / "arquivo_treino.csv"
)

arquivo_modelo_balanceado = (
    pasta
    / "modelos"
    / "random_forest_balanceado.joblib"
)

arquivo_modelo_natural = (
    pasta
    / "modelos"
    / "random_forest_natural.joblib"
)

arquivo_pesos = (
    pasta
    / "modelos"
    / "pesos_combinacao.json"
)


# ============================================================
# CARREGAR DADOS
# ============================================================

df = pd.read_csv(arquivo_treino)


# ============================================================
# COLUNAS UTILIZADAS PELO MODELO
# ============================================================

colunas_categoricas = [
    "AIRLINE",
    "DESTINATION_AIRPORT",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR"
]

colunas_numericas = [
    "DAY",
    "DEPARTURE_MINUTES",
    "SCHEDULED_TIME",
    "DISTANCE"
]

colunas_modelo = (
    colunas_categoricas
    + colunas_numericas
)


# ============================================================
# X E y
# ============================================================

X = df[colunas_modelo].copy()

y = df["DELAYED"].copy()


# ============================================================
# FUNÇÃO PARA CRIAR A PREPARAÇÃO
# ============================================================

def criar_preparacao():

    return ColumnTransformer(
        [
            (
                "categoricas",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                colunas_categoricas
            ),

            (
                "numericas",
                "passthrough",
                colunas_numericas
            )
        ]
    )


# ============================================================
# SEPARAR TREINO E VALIDAÇÃO
# ============================================================

X_treino, X_validacao, y_treino, y_validacao = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==============================")
print("DADOS")
print("==============================")

print("Total:", len(X))
print("Treino:", len(X_treino))
print("Validação:", len(X_validacao))

print(
    "Taxa de atraso no treino:",
    y_treino.mean()
)

print(
    "Taxa de atraso na validação:",
    y_validacao.mean()
)


# ============================================================
# MODELO 1 - BALANCEADO
# ============================================================

modelo_balanceado = Pipeline(
    [
        (
            "preparacao",
            criar_preparacao()
        ),

        (
            "modelo",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_leaf=10,

                class_weight="balanced_subsample",

                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# MODELO 2 - NATURAL
# ============================================================

modelo_natural = Pipeline(
    [
        (
            "preparacao",
            criar_preparacao()
        ),

        (
            "modelo",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_leaf=10,

                class_weight=None,

                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# TREINAR MODELO BALANCEADO
# ============================================================

print("\n==============================")
print("TREINAMENTO")
print("==============================")

print("\nTreinando modelo balanceado...")

modelo_balanceado.fit(
    X_treino,
    y_treino
)

print("Modelo balanceado treinado.")


# ============================================================
# TREINAR MODELO NATURAL
# ============================================================

print("\nTreinando modelo natural...")

modelo_natural.fit(
    X_treino,
    y_treino
)

print("Modelo natural treinado.")


# ============================================================
# PROBABILIDADES NA VALIDAÇÃO
# ============================================================

prob_balanceado = (
    modelo_balanceado
    .predict_proba(X_validacao)[:, 1]
)

prob_natural = (
    modelo_natural
    .predict_proba(X_validacao)[:, 1]
)


# ============================================================
# AVERAGE PRECISION DOS MODELOS INDIVIDUAIS
# ============================================================

ap_balanceado = average_precision_score(
    y_validacao,
    prob_balanceado
)

ap_natural = average_precision_score(
    y_validacao,
    prob_natural
)


print("\n==============================")
print("MODELOS INDIVIDUAIS")
print("==============================")

print(
    f"Average Precision balanceado: "
    f"{ap_balanceado:.6f}"
)

print(
    f"Average Precision natural:    "
    f"{ap_natural:.6f}"
)


# ============================================================
# PROCURAR MELHOR PESO
# ============================================================

melhor_ap = -1.0

melhor_peso_balanceado = 0.0
melhor_peso_natural = 1.0


print("\n==============================")
print("TESTANDO PESOS")
print("==============================")


for i in range(0, 101):

    # ----------------------------------------
    # Pesos
    # ----------------------------------------

    peso_balanceado = i / 100

    peso_natural = (
        1 - peso_balanceado
    )


    # ----------------------------------------
    # Score combinado
    # ----------------------------------------

    prob_combinada = (
        peso_balanceado
        * prob_balanceado
        +
        peso_natural
        * prob_natural
    )


    # ----------------------------------------
    # Average Precision
    # ----------------------------------------

    ap = average_precision_score(
        y_validacao,
        prob_combinada
    )


    # ----------------------------------------
    # Mostrar de 10 em 10%
    # ----------------------------------------

    if i % 10 == 0:

        print(
            f"Balanceado: {peso_balanceado:.2f}"
            f" | Natural: {peso_natural:.2f}"
            f" | AP: {ap:.6f}"
        )


    # ----------------------------------------
    # Guardar melhor resultado
    # ----------------------------------------

    if ap > melhor_ap:

        melhor_ap = ap

        melhor_peso_balanceado = (
            peso_balanceado
        )

        melhor_peso_natural = (
            peso_natural
        )


# ============================================================
# MELHOR COMBINAÇÃO
# ============================================================

print("\n==============================")
print("MELHOR COMBINAÇÃO")
print("==============================")

print(
    f"Peso balanceado: "
    f"{melhor_peso_balanceado:.2f}"
)

print(
    f"Peso natural:    "
    f"{melhor_peso_natural:.2f}"
)

print(
    f"Average Precision combinada: "
    f"{melhor_ap:.6f}"
)


# ============================================================
# COMPARAÇÃO COM OS MODELOS INDIVIDUAIS
# ============================================================

print("\n==============================")
print("COMPARAÇÃO")
print("==============================")

print(
    f"Balanceado: {ap_balanceado:.6f}"
)

print(
    f"Natural:    {ap_natural:.6f}"
)

print(
    f"Combinado:  {melhor_ap:.6f}"
)


# ============================================================
# CRIAR PASTA DOS MODELOS
# ============================================================

arquivo_modelo_balanceado.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SALVAR MODELOS
# ============================================================

dump(
    modelo_balanceado,
    arquivo_modelo_balanceado
)

dump(
    modelo_natural,
    arquivo_modelo_natural
)


# ============================================================
# SALVAR PESOS E RESULTADOS
# ============================================================

pesos = {
    "peso_balanceado": melhor_peso_balanceado,
    "peso_natural": melhor_peso_natural,

    "average_precision_balanceado": ap_balanceado,
    "average_precision_natural": ap_natural,
    "average_precision_combinado": melhor_ap
}


with open(
    arquivo_pesos,
    "w"
) as arquivo:

    json.dump(
        pesos,
        arquivo,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n==============================")
print("TREINAMENTO CONCLUÍDO")
print("==============================")

print("\nModelo balanceado:")
print(arquivo_modelo_balanceado)

print("\nModelo natural:")
print(arquivo_modelo_natural)

print("\nPesos:")
print(arquivo_pesos)