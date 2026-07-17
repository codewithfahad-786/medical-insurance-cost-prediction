# ==================================
# Medical Insurance Cost Prediction
# Streamlit Application
# ==================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Medical Insurance Prediction",
    page_icon="🏥",
    layout="centered"
)


# ==========================
# Load Saved Files
# ==========================

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

columns = joblib.load("columns.pkl")


# ==========================
# Application Title
# ==========================

st.title("🏥 Medical Insurance Cost Prediction")

st.write(
    "Predict medical insurance charges using Machine Learning"
)


st.divider()


# ==========================
# User Inputs
# ==========================

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)


sex = st.selectbox(
    "Gender",
    ["male", "female"]
)


bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)


children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)


smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)


region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)



# ==========================
# Prediction Button
# ==========================

if st.button("Predict Insurance Cost"):


    # Create Input DataFrame

    input_data = pd.DataFrame({

        "age":[age],

        "sex":[sex],

        "bmi":[bmi],

        "children":[children],

        "smoker":[smoker],

        "region":[region]

    })


    # ==========================
    # Encoding
    # ==========================

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "sex",
            "smoker",
            "region"
        ],
        dtype=int
    )


    # Add Missing Columns

    for col in columns:

        if col not in input_data.columns:

            input_data[col] = 0



    # Arrange Columns Same As Training

    input_data = input_data[columns]



    # ==========================
    # Scaling
    # ==========================

    numeric_columns = [
        "age",
        "bmi",
        "children"
    ]


    input_data[numeric_columns] = scaler.transform(
        input_data[numeric_columns]
    )



    # ==========================
    # Prediction
    # ==========================

    prediction = model.predict(
        input_data
    )


    result = prediction[0]


    st.success(
        f"Estimated Insurance Cost: ${result:,.2f}"
    )