# Automobile Price Prediction App

This project is a Machine Learning web application that predicts the price of an automobile based on various vehicle specifications. The application is built using Python and Streamlit, with a trained Random Forest regression model for prediction.

---

## Project Overview

The goal of this project is to demonstrate an end-to-end Machine Learning workflow, starting from data preprocessing and model training to deploying the trained model as an interactive web application.

Users can enter automobile details such as company, body style, engine type, mileage, horsepower, and other technical specifications, and the app will predict the estimated market price of the vehicle.

---

## Features

- Predicts automobile prices using a trained Random Forest model
- Handles both categorical and numerical input features
- Interactive and user-friendly web interface built with Streamlit
- Custom frontend styling using HTML and CSS
- Real-time prediction based on user inputs

---

## Input Features

The application uses the following input features:

Categorical Features:
- Company
- Body Style
- Engine Type
- Number of Cylinders

Numerical Features:
- Wheel Base
- Length
- Horsepower
- Average Mileage

---

## Machine Learning Model

- Algorithm: Random Forest Regressor
- Data preprocessing includes:
  - Label encoding for categorical variables
  - Scaling of numerical features
- Model and preprocessing objects are serialized using Joblib

The following files are used:
- `rf_model.pkl` – trained Random Forest model
- `scaler.pkl` – feature scaler
- `le1.pkl`, `le2.pkl`, `le3.pkl` – label encoders for categorical features

---

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- NumPy
- Joblib
- HTML and CSS
- 
   ```bash
   pip install -r requirements.txt
