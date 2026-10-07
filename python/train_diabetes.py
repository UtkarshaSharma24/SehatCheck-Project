"""Module 2: ML on health data. Predict diabetes risk from numbers (no symptoms needed).

Run:  python python/train_diabetes.py
Creates: data/diabetes.csv (if missing), python/diabetes_model.pkl, data/diabetes_model.json, python/diabetes_results.txt

Real data? Put your own CSV at data/diabetes.csv with columns:
age, bmi, glucose, bp, family, active, diabetes   (diabetes = 0 or 1)
"""
import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "diabetes.csv"
FEATURES = ["age", "bmi", "glucose", "bp", "family", "active"]
LABELS = {"age": "Age", "bmi": "BMI", "glucose": "Fasting glucose", "bp": "Systolic BP",
          "family": "Family history", "active": "Physical activity"}

# 1) Data: use your CSV if present, otherwise make practice data.
if not CSV.exists():
    rng = np.random.default_rng(7)
    n = 3000
    age = np.clip(rng.normal(42, 14, n), 18, 80)
    bmi = np.clip(rng.normal(27, 5, n), 16, 45)
    glucose = np.clip(rng.normal(92 + 0.4 * (bmi - 27) + 0.15 * (age - 42), 14), 65, 220)
    bp = np.clip(rng.normal(118 + 0.3 * (age - 42) + 0.5 * (bmi - 27), 12), 85, 200)
    family = rng.binomial(1, 0.25, n)
    active = rng.binomial(1, 0.5, n)
    logit = -13.0 + 0.04 * age + 0.10 * bmi + 0.06 * glucose + 0.015 * bp + 0.8 * family - 0.6 * active
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))        # outcome is random, like real life
    pd.DataFrame({"age": age.round(), "bmi": bmi.round(1), "glucose": glucose.round(),
                  "bp": bp.round(), "family": family, "active": active, "diabetes": y}).to_csv(CSV, index=False)
df = pd.read_csv(CSV)

# 2) Split, train two models, compare.
X, y = df[FEATURES], df["diabetes"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
forest = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
lines = [f"People with diabetes in data: {y.mean():.0%}", ""]
for name, m in [("Logistic Regression", logreg), ("Random Forest", forest)]:
    p = m.predict_proba(X_te)[:, 1]
    lines.append(f"{name}: accuracy {accuracy_score(y_te, p > 0.5):.1%}, ROC-AUC {roc_auc_score(y_te, p):.3f}")
lines.append("\nLogistic Regression is used in the apps (easy to explain, and it runs in the browser).")
report = "\n".join(lines)
print(report)
(ROOT / "python" / "diabetes_results.txt").write_text(report, encoding="utf-8")

# 3) Save for Streamlit and for the website.
pickle.dump({"model": logreg, "features": FEATURES}, open(ROOT / "python" / "diabetes_model.pkl", "wb"))
sc, lr = logreg[0], logreg[1]
web = {"features": FEATURES, "labels": LABELS, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
       "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
(ROOT / "data" / "diabetes_model.json").write_text(json.dumps(web), encoding="utf-8")
