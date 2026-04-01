# FraudShield-ML 🛡️

Sistema MLOps end-to-end de detección de fraude financiero. Entrenado con 590K transacciones reales del dataset IEEE-CIS, desplegado como API REST y con reentrenamiento automático semanal.

![CI](https://github.com/MickelCruz/Fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)

---

## ¿Por qué este proyecto?

El fraude financiero cuesta millones de dólares anuales a bancos y cooperativas. Una institución que procesa 100,000 transacciones mensuales puede estar perdiendo más de $500,000 USD al mes sin un sistema automatizado de detección.

FraudShield-ML detecta el 85% de los fraudes con solo 483 falsas alarmas por cada 2,304 fraudes reales.

---

## Impacto en el negocio

Para una cooperativa o fintech que procesa 100,000 transacciones mensuales:

| Concepto | Valor mensual (USD) |
|---|---|
| Pérdida sin modelo | $540,000 |
| Fraudes prevenidos con modelo | $447,750 |
| Costo operativo del modelo | $6,830 |
| **Ahorro neto mensual** | **$440,920** |
| **ROI primer año** | **+2,000%** |
| **Payback period** | **< 1 mes** |

---

## Resultados del modelo

| Métrica | Valor |
|---|---|
| AUC-PR | 0.7473 |
| ROC-AUC | 0.9494 |
| F1 Score | 0.7085 |
| Fraudes detectados | 85.3% (1,965 / 2,304) |
| Falsos positivos | 483 |
| Threshold óptimo | 0.3756 |

---

## Stack

LightGBM · Optuna · SHAP · FastAPI · Docker · MLflow · Evidently · Prefect · GitHub Actions

---

## Monitoring y reentrenamiento

Evidently analiza los datos semanalmente. Si el modelo degrada (ROC-AUC < 0.90 o F1 < 0.60), Prefect dispara el reentrenamiento automáticamente sin intervención manual.
```bash
python flows/retrain_flow.py
```

---

## Tests
```bash
pytest tests/test_api.py -v  # 6 tests — todos pasan
```
