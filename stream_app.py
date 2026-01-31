import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Vehicle Value Predictor",
    page_icon="🚗",
    layout="centered"
)

# --- LOAD MODELS ---
@st.cache_resource
def load_artifacts():
    # Make sure these files are in the same directory
    model = pickle.load(open("best_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    name_encoder = pickle.load(open("name_encoder.pkl", "rb"))
    return model, scaler, name_encoder

try:
    model, scaler, name_encoder = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Error: Model files not found. Please ensure best_model.pkl, scaler.pkl, and name_encoder.pkl are in the same folder.")
    st.stop()

# --- APP HEADER ---
st.title("Vehicle Value Predictor")
st.write("Enter the vehicle details below to estimate its selling price.")
st.divider()

# --- INPUT FORM ---
col1, col2 = st.columns(2)

with col1:
    # Searchable Dropdown for Car Name
    car_names = list(name_encoder.classes_)
    car_name_input = st.selectbox("Vehicle Name", car_names, index=None, placeholder="Type to search...")
    
    year = st.number_input("Year of Purchase", min_value=1990, max_value=2030, value=2018)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, value=27000)

with col2:
    present_price = st.number_input("Current Showroom Price (Lakhs)", min_value=0.0, format="%.2f", value=5.59)
    owner = st.number_input("Previous Owners", min_value=0, max_value=10, value=0)

st.write("") # Spacer

col3, col4, col5 = st.columns(3)
with col3:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
with col4:
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
with col5:
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

st.divider()

# --- PREDICTION LOGIC ---
if st.button("Estimate Price", type="primary"):
    if car_name_input is None:
        st.error("Please select a Vehicle Name to proceed.")
    else:
        # 1. Encode Car Name
        car_name_encoded = name_encoder.transform([car_name_input])[0]

        # 2. Manual One-Hot Encoding
        fuel_diesel = 1 if fuel_type == "Diesel" else 0
        fuel_petrol = 1 if fuel_type == "Petrol" else 0
        seller_individual = 1 if seller_type == "Individual" else 0
        trans_manual = 1 if transmission == "Manual" else 0

        # 3. Create DataFrame (Order must match training data)
        data_dict = {
            'Car_Name': [car_name_encoded],
            'Year': [year],
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Owner': [owner],
            'Fuel_Type_Diesel': [fuel_diesel],
            'Fuel_Type_Petrol': [fuel_petrol],
            'Seller_Type_Individual': [seller_individual],
            'Transmission_Manual': [trans_manual]
        }
        
        df = pd.DataFrame(data_dict)

        # 4. Scale
        final_features = scaler.transform(df)

        # 5. Predict
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        # 6. Display Result
        if output < 0:
            st.warning("The estimated value is negative. This vehicle might have low resale potential.")
        else:
            st.success(f"Estimated Selling Price: ₹ {output} Lakhs")