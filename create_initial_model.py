# create_initial_model.py - Reliable Version
# =============================================
# FINAL COMBINED: CREATE LABELS + TRAIN MODEL
# =============================================

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

print("=== NUTRIPREDICT - CREATING LABELS & MODEL ===")

# Load data
df = pd.read_csv('preprocessed_nutri_data.csv')
print(f"Loaded {df.shape[0]} rows")

# === 1. CREATE RISK LABELS (Safe version) ===
rda_thresholds = {
    'Protein (g)': 16.7,
    'Fiber (g)': 8.3,
    'Sodium (mg)': 2.3,
    'Cholesterol (mg)': 0.3
}

for nutrient, threshold in rda_thresholds.items():
    if nutrient in df.columns:
        if 'Sodium' in nutrient or 'Cholesterol' in nutrient:
            df[f'{nutrient}_risk'] = (df[nutrient] > threshold).astype(int)
        else:
            df[f'{nutrient}_risk'] = (df[nutrient] < threshold).astype(int)
        print(f"✅ Created {nutrient}_risk")
    else:
        print(f"⚠️ Column '{nutrient}' not found")

# Check created labels
risk_cols = [col for col in df.columns if '_risk' in col]
print(f"\nRisk columns created: {risk_cols}")

if len(risk_cols) == 0:
    print("❌ No risk labels created. Check your data.")
    exit()

# === 2. TRAIN MODEL ===
features = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 
            'Fiber (g)', 'Sugars (g)', 'Sodium (mg) (g)', 'Cholesterol (mg) (g)']

X = df[[col for col in features if col in df.columns]].fillna(0)
Y = df[risk_cols].astype(int)

print(f"Training with X shape: {X.shape}, Y shape: {Y.shape}")

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X, Y)

# === 3. SAVE TO MLFLOW ===
with mlflow.start_run(run_name="nutri_model_v1"):
    mlflow.sklearn.log_model(model, "nutri_model")
    print("✅ Model successfully saved to MLflow!")

print("\n🎉 You can now run the Streamlit app.")