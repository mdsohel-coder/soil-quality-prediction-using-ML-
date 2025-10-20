import streamlit as st
import joblib
import numpy as np
import pandas as pd


try:
    model = joblib.load('soil_fertility_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'soil_fertility_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Set the page layout
st.set_page_config(page_title="Soil Fertility Prediction", page_icon="🌱", layout="wide")

# Title and header
st.title("🌱 **Soil Fertility Prediction**")
st.markdown("Predict the soil fertility and test the model with predefined test cases.")

# Sidebar for feature inputs
st.sidebar.header("Input Soil Parameters")

# Define feature names
feature_names = ['N', 'P', 'K', 'ph', 'ec', 'oc', 'S', 'zn', 'fe', 'cu', 'Mn', 'B']

# Option to select custom inputs or use test cases
input_mode = st.sidebar.radio("Choose Input Mode", ("Custom Inputs", "Predefined Test Cases"))

if input_mode == "Custom Inputs":
    # Collecting soil parameters from user inputs
    N = st.sidebar.number_input('Nitrogen (N)', min_value=0, value=138, step=1)
    P = st.sidebar.number_input('Phosphorus (P)', min_value=0.0, value=8.6, step=0.1)
    K = st.sidebar.number_input('Potassium (K)', min_value=0, value=560, step=1)
    ph = st.sidebar.number_input('pH Level', min_value=0.0, value=7.46, step=0.01)
    ec = st.sidebar.number_input('Electrical Conductivity (ec)', min_value=0.0, value=0.62, step=0.01)
    oc = st.sidebar.number_input('Organic Carbon (oc)', min_value=0.0, value=0.70, step=0.01)
    S = st.sidebar.number_input('Sulfur (S)', min_value=0.0, value=5.9, step=0.1)
    zn = st.sidebar.number_input('Zinc (Zn)', min_value=0.0, value=0.24, step=0.01)
    fe = st.sidebar.number_input('Iron (Fe)', min_value=0.0, value=0.31, step=0.01)
    cu = st.sidebar.number_input('Copper (Cu)', min_value=0.0, value=0.77, step=0.01)
    Mn = st.sidebar.number_input('Manganese (Mn)', min_value=0.0, value=8.71, step=0.1)
    B = st.sidebar.number_input('Boron (B)', min_value=0.0, value=0.11, step=0.01)

    # Prepare input features as a DataFrame
    input_data = pd.DataFrame([[N, P, K, ph, ec, oc, S, zn, fe, cu, Mn, B]], columns=feature_names)
    st.write("### Input Parameters:")
    st.dataframe(input_data)

else:
    # Predefined test cases
    test_cases = pd.DataFrame([
        [358, 12.5, 581, 11.15, 0.51, 0.27, 4.13, 1.23, 11.03, 1.23, 11.03, 0.36, 2],  # High Fertility
        [333, 7.6, 729, 7.2, 0.61, 0.38, 7.1, 0.24, 2.94, 0.83, 7.63, 0.15, 1],    # Medium Fertility
        [30, 2.0, 100, 5.0, 0.20, 0.20, 1.0, 0.10, 0.10, 0.30, 2.0, 0.05, 0],    # Low Fertility
    ], columns=feature_names + ["Expected Result"])
    st.write("### Predefined Test Cases:")
    st.dataframe(test_cases)

    # Select a test case
    selected_case = st.selectbox("Select a Test Case", test_cases.index)
    input_data = test_cases.iloc[[selected_case]].drop(columns=["Expected Result"])
    expected_result = test_cases.loc[selected_case, "Expected Result"]

# Predict button
if st.button('Predict Soil Fertility'):
    try:
        # Scale the input features
        scaled_features = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(scaled_features)

        # Display the result
        st.markdown("### **Prediction Result**")
        if prediction[0] == 0:
            result = "🔴 **Low Fertility** - The soil quality is not suitable for farming."
        elif prediction[0] == 1:
            result = "🟡 **Moderate Fertility** - The soil can be improved for better crop growth."
        else:
            result = "🟢 **High Fertility** - The soil is excellent for farming!"

        st.success(result)

        # If using predefined test cases, compare results
        if input_mode == "Predefined Test Cases":
            st.write("### **Testing Results**")
            st.write(f"**Expected Result**: {['Low Fertility', 'Medium Fertility', 'High Fertility'][int(expected_result)]}")
            st.write(f"**Actual Result**: {['Low Fertility', 'Medium Fertility', 'High Fertility'][int(prediction[0])]}")
            if prediction[0] == expected_result:
                st.success("Test Case Passed ✅")
            else:
                st.error("Test Case Failed ❌")

    except Exception as e:
        st.error(f"An error occurred: {e}")


