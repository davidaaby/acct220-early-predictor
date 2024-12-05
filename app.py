import streamlit as st
import pandas as pd
import joblib

# Load models and scaler
log_reg = joblib.load('./logistic_model.pkl')
regressor = joblib.load('./regression_model.pkl')
scaler = joblib.load('./scaler.pkl')

st.title("ACCT 220 Early Predictor")

# File Upload
st.header("Upload a CSV File for Bulk Predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)

        # Check required columns
        required_columns = ['Average_Attendance', 'First_Assignment', 'First_Project', 'First_Exam']
        if not all(column in data.columns for column in required_columns):
            st.error(f"The file must contain the following columns: {', '.join(required_columns)}")
        else:
            # Scale the input data
            scaled_data = scaler.transform(data[required_columns])

            # Generate predictions
            risk_predictions = log_reg.predict(scaled_data)
            final_scores = regressor.predict(scaled_data)

            # Add results to the DataFrame
            data['Risk_Status'] = ['At-Risk' if pred == 1 else 'Not At-Risk' for pred in risk_predictions]
            data['Predicted_Final_Score'] = final_scores

            # Display results
            st.success("Predictions generated successfully!")
            st.dataframe(data)

            # Option to download the results as a CSV file
            csv = data.to_csv(index=False)
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"An error occurred: {e}")
