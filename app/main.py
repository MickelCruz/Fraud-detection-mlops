import pandas as pd
import io

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from app.model import predecir, predecir_batch, get_model_info, resultados

app = FastAPI(
    title="FraudShield ML API",
    description="API para detección de fraude en transacciones financieras usando LightGBM optimizado con Optuna",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "modelo": resultados["modelo"],
        "version": "1.0.0"
    }


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    try:
        return predecir(transaction.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
        if "TransactionAmt" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="El CSV debe contener la columna TransactionAmt"
            )
        return predecir_batch(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))