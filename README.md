# 🏥 Medical Insurance Cost Prediction using Machine Learning

## 📌 Project Overview

Medical Insurance Cost Prediction is a Machine Learning project that predicts the estimated insurance cost of a person based on different health and demographic factors.

The goal of this project is to build a regression model that can help insurance companies estimate medical expenses and understand the factors that influence insurance charges.

---

## 🎯 Problem Statement

Insurance companies need accurate predictions of customer medical costs to design better insurance plans and manage risk.

This project uses Machine Learning algorithms to predict insurance charges using customer information.

---

## 📊 Dataset Features

The dataset contains the following features:

* **Age** - Customer age
* **Sex** - Gender of customer
* **BMI** - Body Mass Index
* **Children** - Number of children/dependents
* **Smoker** - Smoking status
* **Region** - Residential region
* **Charges** - Medical insurance cost (Target Variable)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* FastAPI (if used)

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Encoding
5. Data Splitting
6. Model Training
7. Model Evaluation
8. Model Deployment

---

## 🔍 Machine Learning Models

The following regression algorithms were used:

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* (Add other models if you used them)

---

## 📈 Model Evaluation

The model was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

---

## 🚀 Deployment

The trained model was deployed using:

* Streamlit Web Application
* FastAPI Backend (optional)

Users can enter their details and get a predicted medical insurance cost.

---

## 📂 Project Structure

```
Medical-Insurance-Cost-Prediction/
│
├── app.py                 # Streamlit application
├── main.py                # FastAPI backend (optional)
├── train_model.py         # Model training code
├── model.pkl              # Trained ML model
├── dataset.csv            # Dataset
├── requirements.txt       # Required libraries
└── README.md              # Project documentation
```

---

## ▶️ How to Run Locally

### 1. Clone Repository

```bash
git clone your-repository-link
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

For Streamlit:

```bash
streamlit run app.py
```

For FastAPI:

```bash
uvicorn main:app --reload
```

---

## 💡 Real World Applications

* Insurance premium estimation
* Risk analysis
* Customer cost prediction
* Healthcare analytics

---

## 👨‍💻 Author

**Muhammad Fahad**

Machine Learning Engineer | Python | Scikit-learn | FastAPI | Streamlit

---

## ⭐ Future Improvements

* Add Deep Learning models
* Improve prediction accuracy
* Add cloud deployment
* Add more healthcare features
