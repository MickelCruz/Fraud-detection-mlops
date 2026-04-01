from pydantic import BaseModel, ConfigDict
from typing import Optional


class TransactionInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "TransactionAmt": 150.0,
            "ProductCD":      "W",
            "card4":          "visa",
            "card6":          "credit",
            "P_emaildomain":  "gmail.com",
            "DeviceType":     "desktop",
            "TransactionDT":  86400
        }
    })

    TransactionAmt: float
    ProductCD:      Optional[str]   = None
    card1:          Optional[float] = None
    card2:          Optional[float] = None
    card3:          Optional[float] = None
    card4:          Optional[str]   = None
    card5:          Optional[float] = None
    card6:          Optional[str]   = None
    addr1:          Optional[float] = None
    addr2:          Optional[float] = None
    dist1:          Optional[float] = None
    P_emaildomain:  Optional[str]   = None
    R_emaildomain:  Optional[str]   = None
    DeviceType:     Optional[str]   = None
    DeviceInfo:     Optional[str]   = None
    TransactionDT:  Optional[int]   = None


class PredictionResponse(BaseModel):
    fraude:              bool
    probabilidad_fraude: float
    riesgo:              str
    threshold_usado:     float


class BatchPredictionResponse(BaseModel):
    total_transacciones: int
    fraudes_detectados:  int
    porcentaje_fraude:   float
    predicciones:        list


class HealthResponse(BaseModel):
    status:  str
    modelo:  str
    version: str


class ModelInfoResponse(BaseModel):
    modelo:                  str
    auc_pr:                  float
    roc_auc:                 float
    f1_optimizado:           float
    threshold_optimo:        float
    fraudes_detectados_test: int
    falsos_positivos_test:   int