"""Step 2: Streamlit app with both modules.

Run:  streamlit run python/app.py
"""
import json, pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from pdf_report import make_pdf

ROOT = Path(__file__).resolve().parent.parent
bundle = pickle.load(open(ROOT / "python" / "model.pkl", "rb"))
model, symptoms = bundle["model"], bundle["symptoms"]
table = json.loads((ROOT / "data" / "disease_table.json").read_text(encoding="utf-8"))
labels = table["symptoms"]                       # id -> nice name
by_label = {v: k for k, v in labels.items()}

st.set_page_config(page_title="SehatCheck", page_icon="🩺")
st.title("🩺 SehatCheck")
st.caption("AI symptom checker (Naive Bayes). Learning project, not a medical diagnosis.")

tab1, tab2 = st.tabs(["Symptom checker", "Diabetes risk check"])
dm = pickle.load(open(ROOT / "python" / "diabetes_model.pkl", "rb"))

with tab1:
    col1, col2 = st.columns(2)
    age = col1.number_input("Age", 0, 110, 25)
    days = col2.selectbox("Symptoms since", ["Today", "2-3 days", "4-6 days", "A week or more"], 1)
    chosen = st.multiselect("Select your symptoms (at least 2)", sorted(by_label))
    ids = [by_label[c] for c in chosen]

    if st.button("Check", type="primary"):
        if len(ids) < 2:
            st.warning("Please select at least 2 symptoms.")
        else:
            if "chest" in ids or "breath" in ids:
                st.error("Chest pain or trouble breathing can be serious. Get medical help today.")
            if ("fever" in ids or "highfever" in ids) and (days in ("4-6 days", "A week or more") or age < 5 or age > 60):
                st.warning("Fever in this situation should be checked by a doctor within 1-2 days.")
            x = pd.DataFrame([[int(s in ids) for s in symptoms]], columns=symptoms)
            proba = pd.Series(model.predict_proba(x)[0], index=model.classes_).sort_values(ascending=False)
            st.subheader("Most likely conditions")
            for name, p in proba.head(4).items():
                st.write(f"**{name}** - {p:.0%}")
                st.progress(float(p))
            st.info(table["diseases"][proba.index[0]]["tip"])
            st.caption("Not a diagnosis. If you feel seriously unwell, see a doctor.")
            report = [f"Age: {age}   |   Symptoms started: {days}", "Symptoms: " + ", ".join(chosen), "", "Most likely conditions:"]
            report += [f"  {n}: {p:.0%}" for n, p in proba.head(4).items()]
            report += ["", "Advice: " + table["diseases"][proba.index[0]]["tip"]]
            st.download_button("Download PDF report", make_pdf("Symptom check report", report), "sehatcheck-report.pdf", "application/pdf")

with tab2:
    st.write("No symptoms needed. Enter numbers from a recent health check-up.")
    c1, c2, c3 = st.columns(3)
    d_age = c1.number_input("Age", 18, 90, 45)
    d_bmi = c2.number_input("BMI", 14.0, 60.0, 27.0)
    d_glu = c3.number_input("Fasting glucose (mg/dL)", 50, 400, 100)
    c4, c5, c6 = st.columns(3)
    d_bp = c4.number_input("Systolic BP", 80, 220, 120)
    d_fam = c5.selectbox("Family history of diabetes?", ["No", "Yes"])
    d_act = c6.selectbox("Exercise most days?", ["Yes", "No"])
    if st.button("Check my risk", type="primary", key="dm_btn"):
        row = pd.DataFrame([[d_age, d_bmi, d_glu, d_bp, int(d_fam == "Yes"), int(d_act == "Yes")]], columns=dm["features"])
        risk = float(dm["model"].predict_proba(row)[0][1])
        st.metric("Estimated diabetes risk", f"{risk:.0%}")
        st.progress(risk)
        if risk < 0.2:
            st.success("Low risk. Keep healthy habits and get a yearly check-up.")
        elif risk < 0.5:
            st.warning("Moderate risk. Ask a doctor for a fasting glucose or HbA1c test.")
        else:
            st.error("High risk. Please see a doctor soon for an HbA1c test.")
        st.caption("Trained on practice data, not real patients. Only a blood test can confirm diabetes.")
        lines = [f"Estimated diabetes risk: {risk:.0%}", "",
                 f"Inputs: age {d_age}, BMI {d_bmi}, fasting glucose {d_glu} mg/dL, systolic BP {d_bp} mmHg, "
                 f"family history {d_fam}, exercise {d_act}", "", "Only a blood test (fasting glucose or HbA1c) can confirm diabetes."]
        st.download_button("Download PDF report", make_pdf("Diabetes risk report", lines), "diabetes-risk-report.pdf", "application/pdf", key="dm_pdf")
