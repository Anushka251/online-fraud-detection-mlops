
from fastapi.testclient import TestClient

from api import main
def test_health_endpoint():
    with TestClient(main.app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert "model_loaded" in response.json()


def test_predict_without_model_returns_service_unavailable():
    with TestClient(main.app) as client:
        original_model = main._model
        original_features = main._features

        try:
            main._model = None
            main._features = None

            response = client.post(
                "/predict",
                json={
                    "step": 1,
                    "type": "PAYMENT",
                    "amount": 9839.64,
                    "oldbalanceOrg": 170136.0,
                    "newbalanceOrig": 160296.36,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                },
            )

            assert response.status_code == 503

        finally:
            main._model = original_model
            main._features = original_features


def test_predict_rejects_missing_fields():
    with TestClient(main.app) as client:
        if main._model is None or main._features is None:
            return

        response = client.post(
            "/predict",
            json={
                "step": 1,
                "type": "PAYMENT",
                "amount": 9839.64,
            },
        )

        assert response.status_code == 422
        assert "missing_features" in response.json()["detail"]


def test_predict_with_complete_transaction():
    with TestClient(main.app) as client:
        if main._model is None or main._features is None:
            raise AssertionError(
                "Model artifacts were not loaded. "
                "Check models/smote_rf_azure/."
            )

        # Build a valid request from the actual feature list.
        sample = {
            "step": 1,
            "type": "PAYMENT",
            "amount": 9839.64,
            "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        }

        response = client.post("/predict", json=sample)

        assert response.status_code == 200

        result = response.json()
        assert result["prediction"] in (0, 1)
        assert 0 <= result["fraud_probability"] <= 1
        assert result["label"] in ("fraud", "legitimate")
