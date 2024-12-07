import streamlit as st
import numpy as np
import joblib

# Load the trained model
MODEL_PATH = "health_regression_model.pkl"

@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Function to predict health score
def predict_health_score(height, weight):
    input_data = np.array([[height, weight]])
    predicted_score = model.predict(input_data)[0]
    return predicted_score

# Streamlit app layout
st.title("Health Score Prediction App")
st.write("""
This app predicts a **Health Score** based on your **Height** and **Weight** using a regression model.
""")

# User input for height and weight
height = st.number_input("Enter your height (in cm):", min_value=100.0, max_value=250.0, value=170.0, step=1.0)
weight = st.number_input("Enter your weight (in kg):", min_value=30.0, max_value=200.0, value=70.0, step=1.0)

# Button to make the prediction
if st.button("Predict Health Score"):
    # Predict the health score
    score = predict_health_score(height, weight)
    st.write(f"### Predicted Health Score: **{score:.2f}**")

    # Interpretation of the health score
    if score >= 90:
        st.success("You are likely to be healthy. 🌟")
    elif 50 <= score < 90:
        st.warning("You are moderately healthy. 🧐")
    else:
        st.error("You may need to improve your health. 🩺")

# Additional info
st.info("This prediction is based on a simulated regression model and should not replace professional medical advice.")
