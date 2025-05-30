import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("Predict CO₂ Emissions from Electricity Usage")
st.markdown("This app uses Linear Regression Model to predict CO₂ emissions from electricity usage uploaded via sheets")

uploaded_file = st.file_uploader("Upload a spreadsheet with CO₂ emissions and electricity usage", type="xlsx")

if uploaded_file:
    data = pd.read_excel(uploaded_file)
    st.subheader("Data Used for Training: ")
    st.dataframe(data) #show information as a table
    input_variables=data["kWh"].values
    actual_output=data["emissions"].values

    slope = 0  # m in y = mx + b
    intercept = 0  # b in y = mx + b

    learning_rate = 0.00000001
    num_iterations = 200

    error_history = []  # track the error at each step

    for step in range(num_iterations):
        predicted_output = slope * input_variables + intercept

        error = (actual_output - predicted_output)

        slope_gradient = (2 / len(input_variables)) * np.dot(input_variables, error)
        intercept_gradient = (2 / len(input_variables)) * np.sum(error)

        slope += learning_rate * slope_gradient
        intercept += learning_rate * intercept_gradient

        mse = np.mean(error ** 2)
        error_history.append(mse)


    st.subheader("Trained Linear Model")
    st.write(f"Predicted CO₂ usage: Learned Line y = {round(slope, 2)}x + intercept{round(intercept,2)}")

    st.subheader("try it yourself: ")
    user_kwh = st.number_input("Enter your electricity usage in kWh")
    predicted_co2 = slope * user_kwh + intercept
    print(predicted_co2)
    st.success(f"Estimated CO₂ Emissions: {round(predicted_co2, 2)}")

