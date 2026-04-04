# create_new_logs.py - Generate dummy new_logs.csv for retraining test

import pandas as pd
import numpy as np

print("=== CREATING DUMMY NEW LOGS FOR RETRAINING TEST ===")

# Load existing preprocessed data
df = pd.read_csv('preprocessed_nutri_data.csv')
print(f"Original data shape: {df.shape}")

# Take 50 random rows
new_logs = df.sample(n=50, random_state=42).copy()

# Add realistic noise (simulate new user data)
np.random.seed(42)

# Add noise to numeric columns
numeric_cols = ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 
                'Fiber (g)', 'Sugars (g)', 'Sodium (mg) (g)', 'Cholesterol (mg) (g)']

for col in numeric_cols:
    if col in new_logs.columns:
        noise = np.random.normal(0, 0.08, size=len(new_logs))  # ±8% noise
        new_logs[col] = (new_logs[col] * (1 + noise)).round(1)

# Re-create risk labels on the new data (important!)
rda = {'Protein (g)': 16.7, 'Fiber (g)': 8.3}
for nut, th in rda.items():
    if nut in new_logs.columns:
        new_logs[f'{nut}_risk'] = (new_logs[nut] < th).astype(int)

print(f"New logs created with shape: {new_logs.shape}")
print("Risk columns in new_logs:", [col for col in new_logs.columns if '_risk' in col])

# Save as new_logs.csv
new_logs.to_csv('new_logs.csv', index=False)
print("✅ Saved 'new_logs.csv' successfully!")
print("You can now use this file for retraining.")