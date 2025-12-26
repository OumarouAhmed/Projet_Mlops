import os
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PRODUCTION_MODEL_PATH


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="30 numerical features")


app = FastAPI(title="Breast Cancer Prediction API")


def load_model():
    model_path = Path(os.getenv("MODEL_PATH", str(PRODUCTION_MODEL_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train and export a model first."
        )
    return joblib.load(model_path)


def load_model_version(model_path: Path) -> str:
    version_path = model_path.parent / "VERSION"
    if version_path.exists():
        return version_path.read_text(encoding="utf-8").strip()
    return os.getenv("MODEL_VERSION", "unknown")


@app.on_event("startup")
def startup():
    model_path = Path(os.getenv("MODEL_PATH", str(PRODUCTION_MODEL_PATH)))
    app.state.model = load_model()
    app.state.model_version = load_model_version(model_path)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.model_version}


@app.post("/predict")
def predict(request: PredictRequest):
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features.")
    model = app.state.model
    pred = model.predict([request.features])[0]
    label = "malignant" if int(pred) == 1 else "benign"
    return {"prediction": int(pred), "label": label, "model_version": app.state.model_version}
