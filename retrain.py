# retrain_model.py - Triggered if drift or new data
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load & Append New Data (e.g., from user uploads)
df_old = pd.read_csv('preprocessed_nutri_data.csv')
df_new = pd.read_csv('new_logs.csv')  # Assume new file pushed to repo
df_combined = pd.concat([df_old, df_new], ignore_index=True)

# Setup X/Y (from Obj3)
# ... (numeric_features, target_cols, available_features, available_targets as before)
X = df_combined[available_features].fillna(0)
Y = df_combined[available_targets].astype(int)

# Split & Train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))  # Bump estimators
model.fit(X_train, Y_train)

# Evaluate & Log
Y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
with mlflow.start_run(run_name="retrained_model"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("data_size", len(df_combined))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "nutri_rf_model")
    print(f"✅ Retrained! New Accuracy: {acc:.2%} (Data size: {len(df_combined)})")

# Save updated data
df_combined.to_csv('preprocessed_nutri_data.csv', index=False)

# Run: python retrain_model.py (after adding new_logs.csv)