import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

num_cols  = ["TransactionAmt", "card1", "card2", "card3", "card5",
             "addr1", "addr2", "dist1", "log_amt", "hora", "dia_semana"]
cat_low   = ["ProductCD", "card4", "card6", "DeviceType"]
cat_high  = ["P_emaildomain", "R_emaildomain", "DeviceInfo"]
all_cols  = num_cols + cat_low + cat_high

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ]), num_cols),
    ("cat_low", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), cat_low),
    ("cat_high", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), cat_high),
])

np.random.seed(42)
n = 100
X = pd.DataFrame({
    **{c: np.random.randn(n)        for c in num_cols},
    **{c: ["A", "B"] * (n // 2)    for c in cat_low},
    **{c: ["gmail.com"] * n         for c in cat_high},
})
y = np.random.randint(0, 2, n)

preprocessor.fit(X, y)
X_prep = preprocessor.transform(X)

model = LGBMClassifier(n_estimators=5, random_state=42, verbose=-1)
model.fit(X_prep, y)

joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
joblib.dump(model,        MODELS_DIR / "LightGBM_best.joblib")

resultados = {
    "modelo":                  "LightGBM",
    "auc_pr":                  0.75,
    "roc_auc":                 0.95,
    "f1_optimizado":           0.71,
    "threshold_optimo":        0.3756,
    "fraudes_detectados_test": 1965,
    "falsos_positivos_test":   483
}

with open(MODELS_DIR / "resultados_finales.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("Modelos mock creados correctamente")