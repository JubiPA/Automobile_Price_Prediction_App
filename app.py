import streamlit as st
import joblib
import numpy as np
from word2number import w2n

# Page configuration
st.set_page_config(
    page_title="Automobile Price Prediction",
    layout="centered"
)


# Custom HTML + CSS
st.markdown("""
<style>

/* PAGE BACKGROUND */
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(
            135deg,
            rgba(245, 230, 215, 0.92),
            rgba(230, 210, 190, 0.92)
        ),
        url("https://images.unsplash.com/photo-1503376780353-7e6692767b70");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* MAIN CARD */
.app-container {
    background: rgba(255, 255, 255, 0.88);
    border-radius: 22px;
    padding: 45px;
    box-shadow: 0 20px 45px rgba(90, 60, 30, 0.25);
}

/* TITLE */
.app-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #3b2f2f;
    margin-bottom: 35px;
    letter-spacing: 1px;
}

/* SECTION HEADERS */
.section-title {
    font-size: 18px;
    font-weight: 800;
    color: #4a2c2a;
    margin-top: 35px;
    margin-bottom: 12px;
    text-transform: uppercase;
    border-bottom: 2px solid #c08457;
    padding-bottom: 6px;
}

/* LABELS */
label {
    color: #3b2f2f !important;
    font-weight: 700 !important;
    text-transform: uppercase;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #c08457, #8b5a2b);
    color: white;
    padding: 15px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 800;
    border: none;
    margin-top: 35px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(139, 90, 43, 0.45);
    transition: 0.2s ease;
}

/* RESULT BOX */
.result-box {
    background: linear-gradient(135deg, #e9c9a5, #d2a679);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    margin-top: 30px;
    font-size: 22px;
    font-weight: 800;
    color: #3b2f2f;
}

/* Remove Streamlit footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le1 = joblib.load("le1.pkl")
    le2 = joblib.load("le2.pkl")
    le3 = joblib.load("le3.pkl")
    return model, scaler, le1, le2, le3
model, scaler, le1, le2, le3 = load_model()

# UI Layout
st.markdown("<div class='app-container'>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-title'>AUTOMOBILE PRICE PREDICTION</div>",
    unsafe_allow_html=True
)

# Inputs
st.markdown("<div class='section-title'>Vehicle Information</div>", unsafe_allow_html=True)

company = st.selectbox(
    "Company",
    ["", "alfa-romero", "audi", "bmw", "chevrolet", "dodge", "honda",
     "isuzu", "jaguar", "mazda", "mercedes-benz", "mitsubishi",
     "nissan", "porsche", "toyota", "volkswagen", "volvo"])

body_style = st.selectbox(
    "Body Style",
    ["", "convertible", "hatchback", "sedan", "wagon", "hardtop"])

engine_type = st.selectbox(
    "Engine Type",
    ["", "dohc", "ohcv", "ohc", "l", "rotor", "ohcf", "dohcv"])

num_cylinders = st.selectbox(
    "Number of Cylinders",
    ["", "two", "three", "four", "five", "six", "eight", "twelve"])

st.markdown("<div class='section-title'>Technical Specifications</div>", unsafe_allow_html=True)

wheel_base = st.number_input("Wheel Base", 80.0, 130.0)
length = st.number_input("Length", 140.0, 210.0)
horsepower = st.number_input("Horsepower", 40, 300)
avg_mileage = st.number_input("Average Mileage", 10, 50)

# Prediction
if st.button("Predict Vehicle Price"):
    if "" in [company, body_style, engine_type, num_cylinders]:
        st.error("PLEASE SELECT ALL CATEGORICAL FIELDS.")
    else:
        company_enc = le1.transform([company])[0]
        body_enc = le2.transform([body_style])[0]
        engine_enc = le3.transform([engine_type])[0]
        cyl_enc = w2n.word_to_num(num_cylinders)

        input_data = np.array([[
            company_enc,
            body_enc,
            wheel_base,
            length,
            engine_enc,
            cyl_enc,
            horsepower,
            avg_mileage
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown(
            f"<div class='result-box'>ESTIMATED PRICE : ${prediction:,.2f}</div>",
            unsafe_allow_html=True
        )


st.markdown("</div>", unsafe_allow_html=True)