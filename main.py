"""FastAPI inference app for AgriBot model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

APP_TITLE = "AgriBot Crop Recommendation API"
MODEL_PATH = Path("artifacts/agribot_model.pkl")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

app = FastAPI(title=APP_TITLE)
templates = Jinja2Templates(directory="templates")
_model: Any | None = None


def get_model() -> Any:
    """Load model once and return cached object."""
    global _model  # pylint: disable=global-statement
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Download CI artifact (agribot-model-pickle or inference bundle) and place it under artifacts/."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Render simple HTML form for prediction."""
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "error": None})


@app.post("/", response_class=HTMLResponse)
def predict_form(
    request: Request,
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
) -> HTMLResponse:
    """Predict crop label from form values."""
    try:
        model = get_model()
        row = pd.DataFrame([
            {
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }
        ])
        prediction = str(model.predict(row)[0])
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "error": None})
    except Exception as exc:  # pylint: disable=broad-except
        return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "error": str(exc)})


@app.post("/predict-json")
def predict_json(payload: dict[str, float]) -> dict[str, str]:
    """Predict using JSON payload with feature values."""
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    model = get_model()
    row = pd.DataFrame([{f: payload[f] for f in FEATURES}])
    pred = str(model.predict(row)[0])
    return {"prediction": pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
