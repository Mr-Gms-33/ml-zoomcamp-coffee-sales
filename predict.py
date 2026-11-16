"""Flask service for coffee drink predictions."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from data_utils import FEATURE_COLUMNS

APP = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "pipeline.joblib"
PIPELINE = joblib.load(MODEL_PATH)


def row_from_payload(payload: dict) -> pd.DataFrame:
    cash_type = str(payload.get("cash_type", "card")).strip().lower()
    if cash_type not in {"card", "cash"}:
        raise ValueError("cash_type must be 'card' or 'cash'")
    has_card = "yes" if cash_type == "card" else "no"
    if "has_card" in payload:
        has_card = "yes" if payload["has_card"] in (True, "yes", 1, "1") else "no"

    row = {
        "hour": int(payload["hour"]),
        "day_of_week": int(payload["day_of_week"]),
        "month": int(payload["month"]),
        "is_weekend": int(payload.get("is_weekend", int(payload["day_of_week"]) >= 5)),
        "cash_type": cash_type,
        "has_card": has_card,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@APP.get("/")
def health():
    return jsonify({"status": "ok", "model": str(MODEL_PATH.name)})


@APP.post("/predict")
def predict():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        frame = row_from_payload(payload)
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    proba = PIPELINE.predict_proba(frame)[0]
    classes = PIPELINE.classes_
    probabilities = {label: float(p) for label, p in zip(classes, proba)}
    label = PIPELINE.predict(frame)[0]

    return jsonify({"coffee_name": label, "probabilities": probabilities})


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "9696"))
    APP.run(host=host, port=port)


if __name__ == "__main__":
    main()