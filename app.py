import streamlit as st
import joblib
import numpy as np

# Set page title
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️")

st.title("Breast Cancer Prediction System")
st.write("Enter tumor features to predict whether it is Benign or Malignant.")

# Load the trained model
model = joblib.load("model/breast_cancer_model.pkl")

# Input fields for the 5 features
radius = st.number_input("Radius mean", min_value=0.0, step=0.01)
texture = st.number_input("Texture mean", min_value=0.0, step=0.01)
perimeter = st.number_input("Perimeter mean", min_value=0.0, step=0.01)
area = st.number_input("Area mean", min_value=0.0, step=0.01)
smoothness = st.number_input("Smoothness mean", min_value=0.0, step=0.001)

# Prediction button
if st.button("Predict"):
    features = np.array([[radius, texture, perimeter, area, smoothness]])
    prediction = model.predict(features)
    
    if prediction[0] == 0:
        st.success("Prediction: Benign ✅")
    else:
        st.error("Prediction: Malignant ❌")
