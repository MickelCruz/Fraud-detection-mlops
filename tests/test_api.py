from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelo"] == "LightGBM"
    assert data["version"] == "1.0.0"


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "auc_pr" in data
    assert "roc_auc" in data
    assert "threshold_optimo" in data
    assert data["modelo"] == "LightGBM"


def test_predict_transaccion_normal():
    response = client.post("/predict", json={
        "TransactionAmt": 50.0,
        "card4":          "visa",
        "card6":          "debit",
        "P_emaildomain":  "gmail.com",
        "TransactionDT":  86400
    })
    assert response.status_code == 200
    data = response.json()
    assert "fraude" in data
    assert "probabilidad_fraude" in data
    assert "riesgo" in data
    assert "threshold_usado" in data
    assert data["riesgo"] in ["ALTO", "MEDIO", "BAJO"]


def test_predict_solo_campo_obligatorio():
    response = client.post("/predict", json={
        "TransactionAmt": 100.0
    })
    assert response.status_code == 200
    data = response.json()
    assert "fraude" in data


def test_predict_input_invalido():
    response = client.post("/predict", json={
        "TransactionAmt": "invalido"
    })
    assert response.status_code == 422


def test_predict_sin_transactionamt():
    response = client.post("/predict", json={
        "card4": "visa"
    })
    assert response.status_code == 422