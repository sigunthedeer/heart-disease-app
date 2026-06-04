import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
# joblib - saves Python objects (like trained models) to a file
# so we can load them later instantly without retraining

# Load and prepare data
df = pd.read_csv('heart_cleveland_upload.csv')

print(df.columns.tolist())
print(df.shape) 

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[features]
y = df['target']


# Scale features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model on ALL data (no test split - we already validated it works)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# Save the model AND the scaler to files
joblib.dump(model, 'heart_model.pkl')
joblib.dump(scaler, 'heart_scaler.pkl')
# .pkl - "pickle" file, a saved Python object
# we save the scaler too because new inputs must be scaled the same way

print("Model and scaler saved!")
print(f"Features in order: {features}")
