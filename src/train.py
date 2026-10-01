from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, clean

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL = ROOT / "models" / "model.joblib"
SEED = 42


def test_metrics(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_pr_auc": average_precision_score(y_test, proba),
        "test_f1": f1_score(y_test, pred),
    }


def main():
    df = clean(pd.read_csv(DATA))
    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)

    # Split first; everything that learns from data is fitted on train only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. Baseline
    baseline = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    baseline_cv = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    baseline.fit(X_train, y_train)

    # 2. Tree ensemble with hyperparameter tuning
    forest = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=SEED)),
    ])
    search = RandomizedSearchCV(
        forest,
        param_distributions={
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_leaf": [1, 3, 5, 10],
            "model__max_features": ["sqrt", 0.5],
        },
        n_iter=15, cv=cv, scoring="roc_auc", random_state=SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)

    models = {
        "Logistic Regression (baseline)": (baseline, baseline_cv),
        "Random Forest (tuned)": (search.best_estimator_, search.best_score_),
    }

    rows = []
    for name, (model, cv_auc) in models.items():
        rows.append({"model": name, "cv_roc_auc": cv_auc, **test_metrics(model, X_test, y_test)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nBest forest settings:", search.best_params_)

    # Pick the winner on cross-validation score, not on the test set
    best_name = max(models, key=lambda name: models[name][1])
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(models[best_name][0], MODEL)
    print(f"\nSaved {best_name} to {MODEL}")


if __name__ == "__main__":
    main()