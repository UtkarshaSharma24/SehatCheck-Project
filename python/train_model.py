"""Step 1: build a practice dataset and train a Naive Bayes model.

Run:  python python/train_model.py
Creates: data/dataset.csv, python/model.pkl, python/results.txt
"""
import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

ROOT = Path(__file__).resolve().parent.parent
table = json.loads((ROOT / "data" / "disease_table.json").read_text(encoding="utf-8"))
symptoms = list(table["symptoms"])          # symptom ids, e.g. "fever"
diseases = table["diseases"]                # name -> {prior, tip, p:{symptom: prob}}
FLOOR = table["floor"]

rng = np.random.default_rng(42)

# 1) Make 5000 fake patients: pick a disease, then switch each symptom on/off by its probability.
names = list(diseases)
priors = np.array([diseases[n]["prior"] for n in names])
priors = priors / priors.sum()
rows = []
for _ in range(5000):
    d = rng.choice(names, p=priors)
    probs = diseases[d]["p"]
    rows.append([int(rng.random() < probs.get(s, FLOOR)) for s in symptoms] + [d])
df = pd.DataFrame(rows, columns=symptoms + ["disease"])
df.to_csv(ROOT / "data" / "dataset.csv", index=False)

# 2) Split into train (80%) and test (20%).
X, y = df[symptoms], df["disease"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3) Train and test.
model = BernoulliNB(alpha=1.0).fit(X_tr, y_tr)
pred = model.predict(X_te)
acc = accuracy_score(y_te, pred)

report = f"Accuracy on test data: {acc:.1%}\n\n"
report += classification_report(y_te, pred, zero_division=0)
report += "\nConfusion matrix (rows = real, columns = predicted)\n"
report += str(pd.DataFrame(confusion_matrix(y_te, pred, labels=model.classes_),
                           index=model.classes_, columns=[c[:6] for c in model.classes_]))
print(report)
(ROOT / "python" / "results.txt").write_text(report, encoding="utf-8")

# 4) Save the model for the app.
with open(ROOT / "python" / "model.pkl", "wb") as f:
    pickle.dump({"model": model, "symptoms": symptoms}, f)
print("\nSaved python/model.pkl")
