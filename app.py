import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the saved model and scaler (runs once when the app starts)
model = joblib.load('heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

# The features in the exact order the model expects them
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


# --- PAGE SETUP ---
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")
# set_page_config - sets the browser tab title, icon, and page width
# layout="wide" - uses the full width of the screen

st.title("❤️ Heart Disease Risk Predictor")
# st.title - big heading at the top of the page
st.markdown("Enter patient details below to estimate heart disease risk and see which factors drive the prediction.")
# st.markdown - writes text (supports markdown formatting)

st.warning(
    "⚠️ **Disclaimer:** This is an educational portfolio project, not a medical "
    "tool. It is trained on a small public dataset (303 patients) and must not be "
    "used for real medical decisions. Always consult a qualified healthcare "
    "professional for any health concerns."
)
# st.warning - displays a yellow attention box

# --- INPUT SECTION ---
st.header("Patient Details")

# st.columns - splits the page into side-by-side columns
col1, col2, col3 = st.columns(3)
# creates 3 columns so inputs aren't all in one long stack

with col1:
    # 'with col1:' - everything indented here goes in the first column
    age = st.slider("Age", 20, 100, 50)
    # st.slider(label, min, max, default) - a draggable slider
    sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)],
                       format_func=lambda x: x[0])[1]
    # st.selectbox - a dropdown menu
    # format_func - shows the readable label but stores the number
    cp = st.selectbox("Chest Pain Type",
                      options=[("Typical angina", 0), ("Atypical angina", 1),
                               ("Non-anginal", 2), ("Asymptomatic", 3)],
                      format_func=lambda x: x[0])[1]
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    
with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                       options=[("No", 0), ("Yes", 1)],
                       format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Resting ECG",
                           options=[("Normal", 0), ("ST-T abnormality", 1),
                                    ("LV hypertrophy", 2)],
                           format_func=lambda x: x[0])[1]
    thalach = st.slider("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina",
                         options=[("No", 0), ("Yes", 1)],
                         format_func=lambda x: x[0])[1]
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    
with col3:
    slope = st.selectbox("Slope of Peak Exercise ST",
                         options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                         format_func=lambda x: x[0])[1]
    ca = st.selectbox("Number of Major Vessels (0-3)",
                      options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia",
                        options=[("Normal", 1), ("Fixed defect", 2),
                                 ("Reversible defect", 3)],
                        format_func=lambda x: x[0])[1]
                        
# --- PREDICTION ---
st.header("Prediction")

if st.button("Predict Risk", type="primary"):
    # st.button - creates a clickable button
    # the code inside this 'if' runs only when the button is clicked

    # Gather inputs in the correct order
    inputs = [age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]

    # Scale the inputs the same way the model was trained
    input_array = np.array(inputs).reshape(1, -1)
    # reshape(1, -1) - turns the list into a single row the model can read
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    # [0][1] - probability of class 1 (heart disease)

    # Show result
    if prediction == 1:
        st.error(f"⚠️ Higher Risk Detected - {probability:.0%} probability of heart disease")
        # st.error - red warning box
    else:
        st.success(f"✓ Lower Risk - {probability:.0%} probability of heart disease")
        # st.success - green success box
        
    st.caption(
    "Reminder: this prediction is for demonstration only and is not medical advice."
)
# st.caption - small grey text, good for fine print


# --- THE EDUCATIONAL PART: which factors drove this prediction ---
    st.subheader("What drove this prediction?")

    # Get the model's coefficients and multiply by this patient's scaled values
    contributions = model.coef_[0] * input_scaled[0]
    # this shows how much each feature pushed THIS specific prediction
    # positive = pushed toward disease, negative = pushed toward healthy

    contrib_series = pd.Series(contributions, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1a7a4a' if c < 0 else '#E87722' for c in contrib_series]
    ax.barh(contrib_series.index, contrib_series.values, color=colors)
    ax.set_title("Factor Contributions (orange = increases risk, green = decreases risk)")
    ax.set_xlabel("Contribution to prediction")
    ax.axvline(x=0, color='black', linewidth=0.8)
    st.pyplot(fig)
    # st.pyplot - displays a matplotlib chart in the app

    st.caption("This shows how each factor influenced THIS patient's prediction specifically.")
    
