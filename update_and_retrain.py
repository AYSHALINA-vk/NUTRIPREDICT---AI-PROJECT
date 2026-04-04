# update_and_retrain.py - One-click retraining
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

print("=== ONE-CLICK UPDATE & RETRAIN ===")

# Create new logs with noise
df = pd.read_csv('preprocessed_nutri_data.csv')
new_logs = df.sample(n=50, random_state=42).copy()

np.random.seed(42)
for col in ['Calories (kcal)', 'Protein (g)', 'Fiber (g)']:
    if col in new_logs.columns:
        new_logs[col] = (new_logs[col] * (1 + np.random.normal(0, 0.08, len(new_logs)))).round(1)

# Add risk labels
new_logs['Protein (g)_risk'] = (new_logs['Protein (g)'] < 16.7).astype(int)
new_logs['Fiber (g)_risk'] = (new_logs['Fiber (g)'] < 8.3).astype(int)

new_logs.to_csv('new_logs.csv', index=False)
print("✅ New logs created")

# Retrain
features = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)', 'Sugars (g)', 'Sodium (mg) (g)', 'Cholesterol (mg) (g)']
X = new_logs[[c for c in features if c in new_logs.columns]].fillna(0)
Y = new_logs[['Protein (g)_risk', 'Fiber (g)_risk']].astype(int)

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X, Y)

with mlflow.start_run(run_name="retrained_with_new_logs"):
    mlflow.sklearn.log_model(model, "nutri_model")
    print("✅ New model logged to MLflow!")

print("Done! Refresh Streamlit app.")