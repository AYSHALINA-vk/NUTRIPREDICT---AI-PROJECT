# monitor_drift.py - Run locally or in CI
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Load model/data (assume from Obj3)
df = pd.read_csv('preprocessed_nutri_data.csv')  # Or new appended data
# ... (X, Y setup from Obj3 Cell 2)

# Simulate new data (e.g., append 100 rows with noise for drift test)
new_data = df.sample(100).copy()
new_data['Protein (g)'] *= 1.2  # Simulate diet shift (higher protein)

X_new = new_data[available_features].fillna(0)
Y_new = new_data[available_targets].astype(int)

# Predict & Score
Y_pred_new = model.predict(X_new)
acc = accuracy_score(Y_new, Y_pred_new)
f1 = f1_score(Y_new, Y_pred_new, average='macro')

print(f"Current Accuracy: {acc:.2%}, F1: {f1:.2%}")

# Log to MLflow
with mlflow.start_run(run_name="drift_check"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    if acc < 0.85:  # Threshold from proposal (target >85%)
        print("🚨 Drift detected! Trigger retrain.")
        mlflow.log_param("action", "retrain")
    else:
        print("✅ No drift—model stable.")

# Run: python monitor_drift.py
# View: mlflow ui (opens localhost:5000)