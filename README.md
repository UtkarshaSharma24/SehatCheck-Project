# SehatCheck: AI Symptom Checker (Early Disease Detection)

Beginner-level student project. User symptoms chunta hai, aur **Naive Bayes** model batata hai kaunsi bimari sabse zyada likely hai. Saath mein "doctor ko kab dikhana hai" wali triage advice bhi milti hai.

> Disclaimer: Yeh learning project hai, medical diagnosis nahi. Probability table hand-made hai, real patient data nahi.

## Folder structure
```
sehatcheck-project/
├── web/index.html            Website version (double-click karke browser mein kholo)
├── python/
│   ├── train_model.py        Dataset banata hai + model train + accuracy nikalta hai
│   ├── train_diabetes.py     Module 2: health numbers se diabetes risk model
│   ├── pdf_report.py         PDF report banane ka helper (fpdf2)
│   ├── app.py                Streamlit app (dono modules, 2 tabs)
│   ├── requirements.txt
│   ├── model.pkl, diabetes_model.pkl   Trained models
│   └── results.txt           Accuracy, precision, recall, confusion matrix
└── data/
    ├── disease_table.json    12 diseases, symptom probabilities, advice
    ├── diabetes.csv          Practice health data (apna real CSV yahan rakh sakte ho)
    ├── diabetes_model.json   Trained model ke numbers (website isse use karti hai)
    └── dataset.csv           5000 practice patients (synthetic)
```

## Kaise chalayein
**A) Website (kuch install nahi karna):** `web/index.html` ko Chrome mein kholo.

**B) Python + Streamlit:**
```
cd sehatcheck-project
pip install -r python/requirements.txt
python python/train_model.py        # symptom model train karo (optional)
python python/train_diabetes.py     # diabetes model train karo (optional)
streamlit run python/app.py         # app khulega browser mein
```

## Kaam kaise karta hai (viva ke liye)
1. `disease_table.json` mein har disease ki **prior** (kitni common hai) aur **P(symptom | disease)** likhi hai.
2. `train_model.py` is table se 5000 nakli patients banata hai, 80% par train aur 20% par test karta hai.
3. **Naive Bayes** Bayes' rule lagata hai: `P(disease | symptoms) ∝ P(disease) × ΠP(symptom | disease)`. "Naive" isliye ki woh maanta hai ki symptoms ek-doosre se independent hain.
4. Sab diseases ke scores ko normalize karke percentage banate hain.
5. **Triage:** chest pain ya saans ki dikkat par red alert, lamba bukhar ya bahut chhote/bade age par doctor ki salah.

## Module 2: Diabetes Risk Check (ML on health data)
Canvas ka "symptoms aane se pehle flag karo" wala hissa. User age, BMI, fasting glucose, BP, family history aur exercise daalta hai. **Logistic Regression** risk % deta hai (Random Forest se compare kiya, `python/diabetes_results.txt` dekho). Result ke saath "kya factors risk badha rahe hain" bhi dikhta hai.

**Canvas ke key metrics** website ke neeche live dikhte hain: Screenings completed, Flagged for a doctor, Confirmed by clinician, aur Flag-to-clinician confirmation rate. Yeh browser mein save hote hain (demo ke liye).

**Real data lagana ho:** `data/diabetes.csv` ko apne CSV se badal do (columns: age, bmi, glucose, bp, family, active, diabetes) aur `train_diabetes.py` dobara chalao. Website ke liye `data/diabetes_model.json` ke numbers `web/index.html` ke `const DM = ...` mein paste kar do.

## PDF report
Website aur Streamlit dono mein "Download PDF report" button hai. Website mein jsPDF library (CDN se) PDF banati hai, Python mein fpdf2.

## Viva ke sawal
- **Accuracy 91% kyun, kya yeh real hai?** Nahi. Data hamare apne table se bana hai, isliye model usi pattern ko seekh leta hai. Real data (jaise Kaggle ka Disease Symptom Prediction dataset) par accuracy kam aa sakti hai.
- **Web aur Python mein fark?** Web version sirf chune hue symptoms ginta hai. Python ka BernoulliNB "na chune hue" symptoms ko bhi "absent" maanta hai, isliye percentages thode alag aa sakte hain.
- **Diabetes model ki accuracy kitni hai?** Practice data par lagbhag 77% (ROC-AUC 0.74). Yeh data bhi nakli hai aur outcome random rakha gaya hai, isliye accuracy 91% wale symptom model se kam aur zyada realistic hai. Real clinic data ke bina clinical claim nahi kar sakte (canvas mein bhi yeh "H:" hypothesis hai).
- **Limitations:** sirf 12 diseases, 26 symptoms, symptoms independent maane gaye, sex/medical history use nahi hui.

## Aage badhane ke idea
- Kaggle ka real dataset lagao aur Naive Bayes ko Decision Tree / Random Forest se compare karo.
- Hindi language support, aur zyada diseases.
- Result history aur PDF report.
