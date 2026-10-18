import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


# ============================================================
# CAMINHOS
# ============================================================

pasta = Path(
    r"C:\Users\patri\Desktop\projeto voos em atraso"
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

pasta_modelos = (
    pasta
    / "modelos"
)

pasta_modelos.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CARREGAR DADOS
# ============================================================

print("\nCarregando dados...")

df_treino = pd.read_csv(
    arquivo_treino
)

df_calibracao = pd.read_csv(
    arquivo_calibracao
)

print(
    "Treino:",
    len(df_treino)
)

print(
    "Calibração:",
    len(df_calibracao)
)


# ============================================================
# TARGET
# ============================================================

y_treino = (
    df_treino["DELAYED"]
    .astype(int)
)

y_calibracao = (
    df_calibracao["DELAYED"]
    .astype(int)
)


# ============================================================
# FEATURES CATEGÓRICAS
# ============================================================

features_categoricas = [
    "AIRLINE",
    "DESTINATION_AIRPORT",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR"
]


# ============================================================
# FEATURES NUMÉRICAS
# ============================================================

features_numericas = [

    # Features antigas
    "DAY",
    "DEPARTURE_MINUTES",
    "SCHEDULED_TIME",
    "DISTANCE",

    # Features operacionais
    "CONGESTIONAMENTO_ATL",

    "QTD_VOOS_COMPANHIA_2H",
    "QTD_ATRASOS_COMPANHIA_2H",
    "TAXA_ATRASO_COMPANHIA_2H",

    "TEM_VOO_ANTERIOR",
    "ATRASO_SAIDA_VOO_ANTERIOR",
    "VOO_ANTERIOR_AINDA_NAO_SAIU"
]


# ============================================================
# TODAS AS FEATURES
# ============================================================

features = (
    features_categoricas
    +
    features_numericas
)


print("\n==============================")
print("FEATURES UTILIZADAS")
print("==============================")

for feature in features:
    print(feature)


# ============================================================
# X TREINO / X CALIBRAÇÃO
# ============================================================

X_treino = (
    df_treino[
        features
    ]
)

X_calibracao = (
    df_calibracao[
        features
    ]
)


# ============================================================
# PIPELINE CATEGÓRICO
# ============================================================

pipeline_categorico = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# PIPELINE NUMÉRICO
# ============================================================
#
# Algumas features possuem NaN.
#
# Exemplo:
# ATRASO_SAIDA_VOO_ANTERIOR
#
# ============================================================

pipeline_numerico = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        )
    ]
)


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

preprocessador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            pipeline_categorico,
            features_categoricas
        ),

        (
            "numericas",
            pipeline_numerico,
            features_numericas
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(

    n_estimators=200,

    max_depth=15,

    min_samples_leaf=10,

    class_weight=None,

    random_state=42,

    n_jobs=-1,

    verbose=1
)


# ============================================================
# PIPELINE COMPLETO
# ============================================================

modelo = Pipeline(
    steps=[
        (
            "preprocessador",
            preprocessador
        ),

        (
            "random_forest",
            random_forest
        )
    ]
)


# ============================================================
# TREINAR
# ============================================================

print("\n")
print("=" * 60)
print("TREINANDO RANDOM FOREST")
print("=" * 60)


modelo.fit(
    X_treino,
    y_treino
)


print("\nTreinamento concluído.")


# ============================================================
# PROBABILIDADES NA CALIBRAÇÃO
# ============================================================

print("\nCalculando probabilidades...")

prob_bruta = (
    modelo
    .predict_proba(
        X_calibracao
    )[:, 1]
)


# ============================================================
# MÉTRICAS ANTES DA CALIBRAÇÃO
# ============================================================

roc_bruto = roc_auc_score(
    y_calibracao,
    prob_bruta
)

ap_bruto = average_precision_score(
    y_calibracao,
    prob_bruta
)

brier_bruto = brier_score_loss(
    y_calibracao,
    prob_bruta
)


print("\n")
print("=" * 60)
print("ANTES DA CALIBRAÇÃO")
print("=" * 60)

print(
    f"ROC AUC:           {roc_bruto:.6f}"
)

print(
    f"Average Precision: {ap_bruto:.6f}"
)

print(
    f"Brier Score:       {brier_bruto:.6f}"
)


# ============================================================
# CALIBRAÇÃO ISOTÔNICA
# ============================================================

print("\nCalibrando probabilidades...")


calibrador = IsotonicRegression(
    y_min=0,
    y_max=1,
    out_of_bounds="clip"
)


calibrador.fit(
    prob_bruta,
    y_calibracao
)


# ============================================================
# PROBABILIDADES CALIBRADAS
# ============================================================

prob_calibrada = (
    calibrador.predict(
        prob_bruta
    )
)


# ============================================================
# MÉTRICAS DEPOIS DA CALIBRAÇÃO
# ============================================================

roc_cal = roc_auc_score(
    y_calibracao,
    prob_calibrada
)

ap_cal = average_precision_score(
    y_calibracao,
    prob_calibrada
)

brier_cal = brier_score_loss(
    y_calibracao,
    prob_calibrada
)


print("\n")
print("=" * 60)
print("DEPOIS DA CALIBRAÇÃO")
print("=" * 60)

print(
    f"ROC AUC:           {roc_cal:.6f}"
)

print(
    f"Average Precision: {ap_cal:.6f}"
)

print(
    f"Brier Score:       {brier_cal:.6f}"
)


# ============================================================
# SALVAR MODELO
# ============================================================

arquivo_modelo = (
    pasta_modelos
    / "random_forest_operacional.joblib"
)

joblib.dump(
    modelo,
    arquivo_modelo
)


# ============================================================
# SALVAR CALIBRADOR
# ============================================================

arquivo_calibrador = (
    pasta_modelos
    / "calibrador_operacional.joblib"
)

joblib.dump(
    calibrador,
    arquivo_calibrador
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("MODELO SALVO")
print("=" * 60)

print(
    "\nRandom Forest:"
)

print(
    arquivo_modelo
)

print(
    "\nCalibrador:"
)

print(
    arquivo_calibrador
)

print("\nPróximo passo:")
print(
    "Testar no arquivo_teste_operacional.csv"
)