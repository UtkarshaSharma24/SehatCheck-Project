# SehatCheck: AI Symptom Checker for Early Disease Detection

SehatCheck is a beginner-level AI/ML student project designed to demonstrate **symptom-based disease prediction** and **early health-risk detection**.

The project has two main modules:

1. **AI Symptom Checker** – Uses a Naive Bayes model to predict the disease that is most likely based on selected symptoms.
2. **Diabetes Risk Check** – Uses Logistic Regression on health-related data to estimate diabetes risk.

The system also provides basic **triage guidance**, such as when a user should consider consulting a doctor.

> **Disclaimer:** SehatCheck is an educational and demonstration project. It is **not a medical diagnostic tool** and should not be used for real medical decisions. The probability table and practice datasets used in this project are not based on real patient records.

---

## Project Overview

SehatCheck demonstrates how machine learning can be used to analyze symptoms and basic health information to generate an early health-risk indication.

The project combines:

* Symptom-based disease prediction
* Naive Bayes classification
* Diabetes risk prediction
* Logistic Regression
* Synthetic/practice datasets
* Basic medical triage guidance
* Streamlit application
* Browser-based web interface
* PDF report generation

The project is intended for **learning, experimentation, and academic demonstration**.

---

## Features

### 1. AI Symptom Checker

Users select their symptoms, and the system predicts which disease is most likely according to the trained Naive Bayes model.

The system currently covers:

* 12 diseases
* 26 symptoms
* Disease-specific symptom probabilities
* Basic probability-based prediction
* Triage recommendations

### 2. Diabetes Risk Check

The second module focuses on early diabetes-risk estimation using health-related information.

Users can provide:

* Age
* BMI
* Fasting glucose
* Blood pressure
* Family history
* Physical activity

A Logistic Regression model then generates an estimated risk percentage.

The system also identifies some factors that may be contributing to the predicted risk.

### 3. Triage Guidance

The application provides basic warning-level guidance.

For example:

* Chest pain or difficulty breathing → Red alert
* Prolonged fever → Doctor consultation recommended
* Very young or elderly users → Additional medical attention may be recommended

This guidance is only for demonstration purposes and is **not a substitute for professional medical advice**.

### 4. PDF Reports

The project supports PDF report generation.

* The web version uses **jsPDF**
* The Python/Streamlit version uses **fpdf2**

---

## Project Structure

```text
sehatcheck-project/
│
├── README.md
│
├── web/
│   └── index.html
│
├── python/
│   ├── train_model.py
│   ├── train_diabetes.py
│   ├── pdf_report.py
│   ├── app.py
│   ├── requirements.txt
│   ├── model.pkl
│   ├── diabetes_model.pkl
│   ├── diabetes_results.txt
│   └── results.txt
│
└── data/
    ├── disease_table.json
    ├── diabetes.csv
    ├── diabetes_model.json
    └── dataset.csv
```

### File Description

| File                          | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| `web/index.html`              | Browser-based version of SehatCheck                         |
| `python/app.py`               | Streamlit application containing both modules               |
| `python/train_model.py`       | Generates the practice dataset and trains the symptom model |
| `python/train_diabetes.py`    | Trains the diabetes risk model                              |
| `python/pdf_report.py`        | Helper for generating PDF reports                           |
| `data/disease_table.json`     | Contains disease priors, symptom probabilities, and advice  |
| `data/dataset.csv`            | Synthetic dataset containing 5,000 practice patients        |
| `data/diabetes.csv`           | Practice health dataset for diabetes-risk modeling          |
| `data/diabetes_model.json`    | Model values used by the web version                        |
| `python/model.pkl`            | Trained symptom prediction model                            |
| `python/diabetes_model.pkl`   | Trained diabetes prediction model                           |
| `python/results.txt`          | Symptom model evaluation results                            |
| `python/diabetes_results.txt` | Diabetes model evaluation results                           |

---

## How to Run the Project

### Option 1: Run the Web Version

The web version does not require Python installation.

Open:

```text
web/index.html
```

in Google Chrome or another modern web browser.

You can simply double-click the file to launch the application.

---

### Option 2: Run the Python + Streamlit Version

First, navigate to the project directory:

```bash
cd sehatcheck-project
```

Install the required Python packages:

```bash
pip install -r python/requirements.txt
```

Train the symptom prediction model:

```bash
python python/train_model.py
```

Train the diabetes-risk model:

```bash
python python/train_diabetes.py
```

Finally, start the Streamlit application:

```bash
streamlit run python/app.py
```

The application will open in your browser.

> Model training commands are optional if the pre-trained model files are already available.

---

# How the Symptom Checker Works

The symptom prediction module is based on the **Naive Bayes** approach.

### Step 1: Disease and Symptom Probabilities

The file:

```text
data/disease_table.json
```

contains:

* Disease priors
* Symptom probabilities
* Basic medical advice

For example, the model uses information similar to:

```text
P(symptom | disease)
```

to estimate how strongly a symptom is associated with a particular disease.

### Step 2: Generate Practice Dataset

The `train_model.py` script uses the probability table to generate approximately **5,000 synthetic patient records**.

The generated data is divided into:

* 80% training data
* 20% testing data

### Step 3: Train the Model

A Naive Bayes classifier is trained using the generated data.

The basic principle is:

```text
P(disease | symptoms)
    ∝
P(disease) × Π P(symptom | disease)
```

The model assumes that symptoms are conditionally independent given the disease.

This independence assumption is the reason the method is called **"Naive" Bayes**.

### Step 4: Generate Prediction

The model calculates scores for the possible diseases and converts them into percentages.

The disease with the highest estimated probability is displayed as the most likely result.

---

# Module 2: Diabetes Risk Check

The second module demonstrates the concept of identifying possible health risks **before obvious symptoms become a diagnosis**.

The user provides health-related information such as:

```text
Age
BMI
Fasting Glucose
Blood Pressure
Family History
Physical Activity
```

The system uses **Logistic Regression** to estimate diabetes risk.

A Random Forest model was also compared during experimentation.

The results can be found in:

```text
python/diabetes_results.txt
```

The practice dataset currently produces approximately:

```text
Accuracy: 77%
ROC-AUC: 0.74
```

These results are based on synthetic/practice data and **must not be interpreted as clinical performance**.

---

# Key Metrics

The application includes demonstration metrics related to the project's Lean Canvas.

These include:

* Screenings completed
* Users flagged for a doctor
* Confirmed by clinician
* Flag-to-clinician confirmation rate

For the demonstration version, these values are stored locally in the browser.

They are intended to demonstrate how the proposed product metrics could work in a future real-world system.

---

# Model Evaluation

## Symptom Prediction Model

The current symptom model reports approximately **91% accuracy** on the generated practice dataset.

However, this number should not be interpreted as real-world medical accuracy.

The reason is that the training and testing data are generated from the project's own hand-created probability table. Therefore, the model is learning patterns generated from the same underlying assumptions.

A real clinical dataset could produce significantly different results.

## Diabetes Risk Model

The diabetes model produces approximately:

```text
Accuracy: 77%
ROC-AUC: 0.74
```

The diabetes dataset is also synthetic/practice data, and the outcome variable is randomly generated.

Therefore, the model does not provide evidence of clinical effectiveness.

---

# Web Version vs Python Version

There is a small difference between the web and Python implementations.

### Web Version

The web application primarily counts the symptoms selected by the user and calculates the corresponding disease probabilities.

### Python Version

The Python implementation uses a **Bernoulli Naive Bayes** approach.

It considers both:

* Selected symptoms
* Symptoms that were not selected

Therefore, the percentages shown by the web and Python versions may differ slightly.

---

# Limitations

The current version has several limitations:

* Only 12 diseases are included.
* The system uses 26 symptoms.
* Symptoms are treated as independent for the Naive Bayes model.
* Sex and detailed medical history are not currently included.
* The datasets are synthetic/practice datasets.
* The probability table is manually created.
* The model has not been clinically validated.
* The system should not be used for real medical diagnosis.
* The reported accuracy values should not be considered clinical performance.

---

# Real-World Data

The diabetes module is designed so that the practice dataset can be replaced with another dataset.

The expected CSV columns are:

```text
age
bmi
glucose
bp
family
active
diabetes
```

After replacing the dataset, retrain the model using:

```bash
python python/train_diabetes.py
```

For the web version, the generated values from:

```text
data/diabetes_model.json
```

can then be updated in:

```text
web/index.html
```

inside the:

```javascript
const DM = ...
```

section.

> Any use of real health data should follow appropriate privacy, security, consent, and regulatory re
