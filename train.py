"""Train final coffee drink classifier and save sklearn pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, get_training_frame

MODELS_DIR = Path(__file__).resolve().parent / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def make_estimator() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("clf", clf)])


def main() -> None:
    x, y = get_training_frame()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    search = RandomizedSearchCV(
        make_estimator(),
        param_distributions={
            "clf__n_estimators": randint(50, 200),
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_leaf": randint(1, 5),
        },
        n_iter=15,
        cv=3,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)

    tuned = search.best_estimator_
    y_pred = tuned.predict(x_test)
    metrics = {
        "best_params": search.best_params_,
        "cv_best_f1_macro": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "feature_columns": FEATURE_COLUMNS,
        "classes": sorted(y.unique().tolist()),
    }

    final = make_estimator()
    final.set_params(**search.best_params_)
    final.fit(x, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, PIPELINE_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {PIPELINE_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()