import streamlit as st
import pickle
import numpy as np

# Load the trained SVM model
with open("svm_model.pkl", "rb") as model_file:
    svm_model = pickle.load(model_file)

# Function to make prediction
def predict_purchase(gender, age, salary):
    # Encode gender: Male = 1, Female = 0
    gender_encoded = 1 if gender == 'Male' else 0

    # Create the input array for prediction
    input_data = np.array([[gender_encoded, age, salary]])
    
    # Predict using the SVM model
    prediction = svm_model.predict(input_data)
    
    # Convert prediction to label
    if prediction == 1:
        return "Yes"
    else:
        return "No"

# Streamlit Interface
st.title("Purchase Prediction App")
st.write("Enter the following details to predict if a person will make a purchase:")

# Input fields
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Enter Age", min_value=0, max_value=100, step=1)
salary = st.number_input("Enter Salary", min_value=0, step=1000)

# Predict button
if st.button("Predict"):
    result = predict_purchase(gender, age, salary)
    st.write(f"Prediction: {result}")

