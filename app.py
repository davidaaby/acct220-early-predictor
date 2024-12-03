import streamlit as st
import joblib
import pandas as pd

# Load models and scaler
log_reg = joblib.load('/content/logistic_model.pkl')
regressor = joblib.load('/content/regression_model.pkl')
scaler = joblib.load('/content/scaler.pkl')

# Streamlit app
st.title("Accounting 220 early Predictors")

# Input fields for the user
attendance = st.number_input("Attendance Percentage (0-100):", min_value=0.0, max_value=100.0, step=1.0)
first_assignment = st.number_input("First Assignment Percentage (0-100):", min_value=0.0, max_value=100.0, step=1.0)
first_project = st.number_input("First Project Percentage (0-100):", min_value=0.0, max_value=100.0, step=1.0)
first_exam = st.number_input("First Exam Percentage (0-100):", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict"):
    try:
        # Prepare input data with corrected column names
        input_data = pd.DataFrame(
            [[attendance, first_assignment, first_project, first_exam]],
            columns=['Average_Attendance', 'First_Assignment', 'First_Project', 'First_Exam']  # Corrected column names
        )
        student_data = scaler.transform(input_data)

        # Make predictions
        risk_prediction = log_reg.predict(student_data)
        predicted_score = regressor.predict(student_data)[0]

        # Display results
        risk_label = "At-Risk" if risk_prediction[0] == 1 else "Not At-Risk"
        st.success(f"Risk Status: {risk_label}")
        st.info(f"Predicted Final Score: {predicted_score:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
