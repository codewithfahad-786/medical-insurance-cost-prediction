# ==================================
# Medical Insurance Prediction API
# FastAPI Backend
# ==================================

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib


# ==========================
# Create FastAPI App
# ==========================

app = FastAPI(
    title="Medical Insurance Prediction API",
    description="Machine Learning API for insurance cost prediction",
    version="1.0"
)


# ==========================
# Load Model Files
# ==========================

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

columns = joblib.load("columns.pkl")


# ==========================
# Input Schema
# ==========================

class InsuranceInput(BaseModel):

    age: int

    sex: str

    bmi: float

    children: int

    smoker: str

    region: str



# ==========================
# Home Route
# ==========================

@app.get("/")
def home():

    return {

        "message":
        "Medical Insurance Prediction API Running"

    }



# ==========================
# Prediction API
# ==========================

@app.post("/predict")
def predict(data: InsuranceInput):


    # Convert Input To DataFrame

    input_data = pd.DataFrame({

        "age":[data.age],

        "sex":[data.sex],

        "bmi":[data.bmi],

        "children":[data.children],

        "smoker":[data.smoker],

        "region":[data.region]

    })



    # Encoding

    input_data = pd.get_dummies(

        input_data,

        columns=[
            "sex",
            "smoker",
            "region"
        ],

        dtype=int

    )



    # Missing Columns Handling

    for col in columns:

        if col not in input_data.columns:

            input_data[col] = 0



    # Same order as training

    input_data = input_data[columns]



    # Scaling

    numeric_columns = [
        "age",
        "bmi",
        "children"
    ]


    input_data[numeric_columns] = scaler.transform(
        input_data[numeric_columns]
    )



    # Prediction

    prediction = model.predict(
        input_data
    )


    return {

        "predicted_insurance_cost":
        float(prediction[0])

    }