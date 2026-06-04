# ❤️ Heart Disease Risk Predictor - Live Interactive App

> An interactive web app that predicts heart disease risk from patient details 
> AND explains which factors drove each prediction. Built with Streamlit.

🔗 **Live app:** https://heart-disease-app-zx68prp8mesde4ce5usc7q.streamlit.app/

## What It Does

Enter a patient's clinical details - age, chest pain type, cholesterol, max heart 
rate, and more - and the app returns a risk prediction along with a chart showing 
exactly which factors pushed the prediction toward higher or lower risk.

## Why It's Different

Most prediction tools are black boxes. This one is **interpretable** - for every 
prediction, it shows the per-patient factor contributions, so you can see *why* 
the model decided what it did. Model interpretability is increasingly essential 
in healthcare ML.

## ⚠️ Disclaimer

This is an **educational portfolio project, not a medical tool**. It is trained on 
a small public dataset (303 patients) and must not be used for real medical 
decisions. Always consult a qualified healthcare professional.

## How It Works

- A Logistic Regression model (91%+ accuracy) trained on the Cleveland Heart 
  Disease dataset, saved with joblib
- A Streamlit interface with sliders and dropdowns for patient inputs
- Per-prediction interpretation via the model's coefficients

## Tools Used

Python · Streamlit · scikit-learn · pandas · matplotlib · joblib

## Run Locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`
