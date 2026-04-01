import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

from prefect import flow, task, get_run_logger
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

# Definir estructura
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset

from prefect.schedules import Cron

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"


@task
def run_monitoring() -> None:
    logger = get_run_logger()

    # Cargar datos y modelo
    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    X_test  = pd.read_parquet(DATA_PROCESSED / "X_test.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(DATA_PROCESSED / "y_test.parquet").squeeze()
    model   = joblib.load(MODELS_DIR / "LightGBM_best.joblib")

    # Recuperar nombres de columnas
    preprocessor  = joblib.load(MODELS_DIR / "preprocessor.joblib")
    num_cols      = preprocessor.transformers_[0][2]
    cat_low       = preprocessor.transformers_[1][2]
    cat_high      = preprocessor.transformers_[2][2]
    feature_names = num_cols + cat_low + cat_high

    X_train.columns = feature_names
    X_test.columns  = feature_names

    # Preparar DataFrames con target y predicciones
    train_df = X_train.sample(n=10000, random_state=42).copy()
    test_df  = X_test.copy()

    train_df["target"]     = y_train.loc[train_df.index].values
    train_df["prediction"] = model.predict_proba(train_df[feature_names])[:, 1]
    test_df["target"]      = y_test.values
    test_df["prediction"]  = model.predict_proba(test_df[feature_names])[:, 1]


    data_definition = DataDefinition(
        classification=[BinaryClassification(target="target", prediction_probas="prediction")]
    )

    ref_dataset  = Dataset.from_pandas(train_df, data_definition=data_definition)
    curr_dataset = Dataset.from_pandas(test_df,  data_definition=data_definition)

    # Ejecutar reporte de clasificación
    report   = Report(metrics=[ClassificationPreset()])
    snapshot = report.run(current_data=curr_dataset, reference_data=ref_dataset)
    results  = snapshot.dict()

    # Extraer métricas
    metricas = {m["metric_name"].split("(")[0]: m["value"] for m in results["metrics"]}
    roc_auc  = metricas["RocAuc"]
    f1       = metricas["F1Score"]

    UMBRAL_ROC_AUC = 0.90
    UMBRAL_F1      = 0.60
    reentrenar     = roc_auc < UMBRAL_ROC_AUC or f1 < UMBRAL_F1

    # Guardar estado
    estado = {
        "fecha":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "roc_auc":         round(roc_auc, 4),
        "f1":              round(f1, 4),
        "umbral_roc_auc":  UMBRAL_ROC_AUC,
        "umbral_f1":       UMBRAL_F1,
        "drift_detectado": False,
        "reentrenar":      reentrenar
    }

    with open(MODELS_DIR / "monitoring_status.json", "w") as f:
        json.dump(estado, f, indent=2)

    logger.info(f"ROC AUC: {roc_auc:.4f} | F1: {f1:.4f} | Reentrenar: {reentrenar}")

@task
def check_monitoring() -> bool:
    logger = get_run_logger()
    with open(MODELS_DIR / "monitoring_status.json") as f:
        status = json.load(f)
    logger.info(f"Estado del monitoring: {status}")
    logger.info(f"Reentrenar: {status['reentrenar']}")
    return status["reentrenar"]


@task
def load_data() -> tuple:
    logger = get_run_logger()
    X_train = pd.read_parquet(DATA_PROCESSED / "X_train.parquet")
    X_val   = pd.read_parquet(DATA_PROCESSED / "X_val.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "y_train.parquet").squeeze()
    y_val   = pd.read_parquet(DATA_PROCESSED / "y_val.parquet").squeeze()
    logger.info(f"Datos cargados — X_train: {X_train.shape} | X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


@task
def train_model(X_train, y_train) -> LGBMClassifier:
    logger = get_run_logger()
    with open(MODELS_DIR / "best_params.json") as f:
        best_params = json.load(f)["LightGBM"]
    model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    logger.info("Modelo reentrenado con best_params.json")
    return model


@task
def evaluate_model(model, X_val, y_val) -> float:
    logger = get_run_logger()
    proba  = model.predict_proba(X_val)[:, 1]
    auc_pr = average_precision_score(y_val, proba)
    logger.info(f"AUC-PR en validación: {auc_pr:.4f}")
    return auc_pr


@task
def save_model(model, auc_pr) -> None:
    logger = get_run_logger()
    with open(MODELS_DIR / "resultados_finales.json") as f:
        resultados = json.load(f)
    auc_pr_actual = resultados["auc_pr"]
    if auc_pr > auc_pr_actual:
        joblib.dump(model, MODELS_DIR / "LightGBM_best.joblib")
        resultados["auc_pr"]  = round(auc_pr, 4)
        resultados["fecha_reentrenamiento"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(MODELS_DIR / "resultados_finales.json", "w") as f:
            json.dump(resultados, f, indent=2)
        logger.info(f"Nuevo modelo guardado — AUC-PR mejoró: {auc_pr_actual:.4f} → {auc_pr:.4f}")
    else:
        logger.info(f"Modelo NO guardado — AUC-PR no mejoró: {auc_pr:.4f} <= {auc_pr_actual:.4f}")


@flow(name="fraudshield-retrain")
def retrain_flow():
    logger = get_run_logger()

    # Paso 1 — ejecutar monitoring y actualizar JSON
    run_monitoring()

    # Paso 2 — leer decisión
    reentrenar = check_monitoring()

    if not reentrenar:
        logger.info("No se requiere reentrenamiento — modelo dentro de parámetros aceptables")
        return

    # Paso 3 — reentrenar
    logger.info("Iniciando reentrenamiento...")
    X_train, X_val, y_train, y_val = load_data()
    model  = train_model(X_train, y_train)
    auc_pr = evaluate_model(model, X_val, y_val)
    save_model(model, auc_pr)
    logger.info("Flujo completado")


if __name__ == "__main__":
    retrain_flow.serve(
        name="fraudshield-retrain-semanal",
        schedule=Cron("0 9 * * 1")  # Cada lunes a las 9am
    )