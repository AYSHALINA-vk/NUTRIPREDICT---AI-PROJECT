# app.py - Streamlit Dashboard with CD Model
import streamlit as st
import mlflow.sklearn
import pandas as pd

# Load Latest Model from MLflow (CD: Auto-updates)
model = mlflow.sklearn.load_model("runs:/latest/nutri_model")

st.title("NutriPredict: AI Deficiency Detector")

# User Input (from Obj3)
calories = st.number_input("Daily Calories (kcal)", 0)
protein = st.number_input("Daily Protein (g)", 0)
# ... (other inputs)

if st.button("Predict Deficits"):
    user_input = pd.DataFrame({
        'Calories (kcal)': [calories],
        'Protein (g)': [protein],
        # ... (align to available_features)
    })
    risks = model.predict(user_input)[0]
    st.write(f"Protein Risk: {'High' if risks[0] == 1 else 'Low'}")
    # Recs from Obj3

# Run: streamlit run app.py