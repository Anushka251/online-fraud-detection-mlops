from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "smote_rf_azure"
MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURES_PATH = MODEL_DIR / "features.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

app = FastAPI(title="Online Fraud Detection API", version="1.0.0")

_model = None
_features = None
_threshold = 0.5


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    label: str


@app.on_event("startup")
def load_model():
    global _model, _features, _threshold

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return

    _model = joblib.load(MODEL_PATH)
    _features = json.loads(
        FEATURES_PATH.read_text(encoding="utf-8")
    )

    if METADATA_PATH.exists():
        metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )
        _threshold = float(metadata.get("threshold", 0.5))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if _model is None or _features is None:
        raise HTTPException(
            status_code=503,
            detail="Model or feature configuration not loaded."
        )

    values = request.model_dump()

    missing = [
        feature for feature in _features
        if feature not in values
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing}
        )

    try:
        row = pd.DataFrame(
            [{feature: values[feature] for feature in _features}],
            columns=_features,
        )

        probability = float(
            _model.predict_proba(row)[0, 1]
        )
        prediction = int(probability >= _threshold)

        return PredictionResponse(
            prediction=prediction,
            fraud_probability=probability,
            label="fraud" if prediction == 1 else "legitimate",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not score input: {exc}",
        ) from exc