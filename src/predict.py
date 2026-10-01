import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from preprocess import clean

MODEL = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


def predict(customer: dict) -> dict:
    model = joblib.load(MODEL)
    df = clean(pd.DataFrame([customer]))
    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(model.predict(df)[0])
    return {
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
    }


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        print(json.dumps(predict(json.load(f)), indent=2))