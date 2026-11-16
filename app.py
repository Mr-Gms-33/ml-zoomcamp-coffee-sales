"""Gradio app for Hugging Face Spaces (cloud bonus)."""

from __future__ import annotations

import joblib
import pandas as pd
import gradio as gr

from data_utils import FEATURE_COLUMNS

PIPELINE = joblib.load("models/pipeline.joblib")
CLASSES = list(PIPELINE.classes_)


def predict_drink(hour: int, day_of_week: int, month: int, cash_type: str) -> tuple[str, dict]:
    cash_type = cash_type.lower().strip()
    row = pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": 1 if day_of_week >= 5 else 0,
                "cash_type": cash_type,
                "has_card": "yes" if cash_type == "card" else "no",
            }
        ]
    )[FEATURE_COLUMNS]
    proba = PIPELINE.predict_proba(row)[0]
    label = PIPELINE.predict(row)[0]
    probs = {c: float(p) for c, p in zip(CLASSES, proba)}
    return label, probs


with gr.Blocks(title="Coffee Sales — Drink Prediction") as demo:
    gr.Markdown(
        "## ML Zoomcamp midterm — predict likely drink\n"
        "Given time context and payment type (machine 1 catalog)."
    )
    with gr.Row():
        hour = gr.Slider(0, 23, value=10, step=1, label="Hour of day")
        dow = gr.Slider(0, 6, value=0, step=1, label="Day of week (0=Mon)")
        month = gr.Slider(1, 12, value=3, step=1, label="Month")
        cash = gr.Radio(["card", "cash"], value="card", label="Payment type")
    predict_btn = gr.Button("Predict")
    drink = gr.Textbox(label="Predicted drink")
    probs = gr.Label(label="Class probabilities", num_top_classes=8)
    predict_btn.click(fn=predict_drink, inputs=[hour, dow, month, cash], outputs=[drink, probs])

if __name__ == "__main__":
    demo.launch()