import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR    = PROJECT_ROOT / "models"

# Cargar modelo, preprocesador y resultados al arrancar la API
model        = joblib.load(MODELS_DIR / "LightGBM_best.joblib")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

with open(MODELS_DIR / "resultados_finales.json") as f:
    resultados = json.load(f)

THRESHOLD = resultados["threshold_optimo"]

# Recuperar nombres de columnas del preprocesador
num_cols      = preprocessor.transformers_[0][2]
cat_low       = preprocessor.transformers_[1][2]
cat_high      = preprocessor.transformers_[2][2]
FEATURE_COLS  = num_cols + cat_low + cat_high

print(f"Modelo cargado: {resultados['modelo']} | Threshold: {THRESHOLD:.4f}")


def preparar_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Features derivadas
    df["log_amt"] = np.log1p(df["TransactionAmt"])
    if "TransactionDT" in df.columns:
        df["hora"]       = (df["TransactionDT"] // 3600) % 24
        df["dia_semana"] = (df["TransactionDT"] // 86400) % 7

    # Agregar columnas faltantes en una sola operación
    cols_faltantes = [col for col in FEATURE_COLS if col not in df.columns]
    df_faltantes   = pd.DataFrame(np.nan, index=df.index, columns=cols_faltantes)
    df             = pd.concat([df, df_faltantes], axis=1)

    return df[FEATURE_COLS]


def predecir(data: dict) -> dict:
    df      = preparar_input(data)
    X       = pd.DataFrame(preprocessor.transform(df), columns=FEATURE_COLS)
    proba   = float(model.predict_proba(X)[0, 1])
    fraude  = proba >= THRESHOLD
    riesgo  = "ALTO" if proba >= 0.7 else "MEDIO" if proba >= 0.4 else "BAJO"

    return {
        "fraude":              fraude,
        "probabilidad_fraude": round(proba, 4),
        "riesgo":              riesgo,
        "threshold_usado":     THRESHOLD
    }


def predecir_batch(df_input: pd.DataFrame) -> dict:
    df_input["log_amt"] = np.log1p(df_input["TransactionAmt"])
    if "TransactionDT" in df_input.columns:
        df_input["hora"]       = (df_input["TransactionDT"] // 3600) % 24
        df_input["dia_semana"] = (df_input["TransactionDT"] // 86400) % 7

    cols_faltantes = [col for col in FEATURE_COLS if col not in df_input.columns]
    df_faltantes   = pd.DataFrame(np.nan, index=df_input.index, columns=cols_faltantes)
    df_input       = pd.concat([df_input, df_faltantes], axis=1)

    X       = pd.DataFrame(preprocessor.transform(df_input[FEATURE_COLS]), columns=FEATURE_COLS)
    probas  = model.predict_proba(X)[:, 1]
    fraudes = (probas >= THRESHOLD).astype(int)

    predicciones = [
        {
            "indice":              int(i),
            "fraude":              bool(fraudes[i]),
            "probabilidad_fraude": round(float(probas[i]), 4),
            "riesgo":              "ALTO" if probas[i] >= 0.7 else "MEDIO" if probas[i] >= 0.4 else "BAJO"
        }
        for i in range(len(probas))
    ]

    return {
        "total_transacciones": len(probas),
        "fraudes_detectados":  int(fraudes.sum()),
        "porcentaje_fraude":   round(float(fraudes.mean() * 100), 2),
        "predicciones":        predicciones
    }


def get_model_info() -> dict:
    return resultados