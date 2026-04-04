# model.py - Fixed & Clean Version

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

print("=== NUTRIPREDICT - LOADING / TRAINING MODEL ===")

# Load data
df = pd.read_csv('preprocessed_nutri_data.csv')
print(f"Loaded {df.shape[0]} rows")

# Create risk columns if missing
risk_cols = [col for col in df.columns if '_risk' in col]

if len(risk_cols) == 0:
    print("Creating risk labels...")
    rda = {'Protein (g)': 16.7, 'Fiber (g)': 8.3}
    for nut, th in rda.items():
        if nut in df.columns:
            df[f'{nut}_risk'] = (df[nut] < th).astype(int)
            print(f"✅ Created {nut}_risk")
    risk_cols = [col for col in df.columns if '_risk' in col]

print(f"Available risk columns: {risk_cols}")

# Features
features = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 
            'Fiber (g)', 'Sugars (g)', 'Sodium (mg) (g)', 'Cholesterol (mg) (g)']

available_features = [col for col in features if col in df.columns]

X = df[available_features].fillna(0)
Y = df[risk_cols].astype(int)

print(f"Training shape -> X: {X.shape}, Y: {Y.shape}")

# Train model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X, Y)

# Save to MLflow
with mlflow.start_run(run_name="nutri_model_v1"):
    mlflow.sklearn.log_model(model, "nutri_model")
    print("✅ Model successfully saved to MLflow!")

print("Model is ready for Streamlit app.")