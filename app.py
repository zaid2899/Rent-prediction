import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('rent_model_tuned.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── Data from your dataset ───────────────────────────────────
# Replace these with your actual unique values from Step 2
CITIES = ['Mumbai', 'Bangalore', 'New Delhi', 'Pune', 'Nagpur']

LOCALITIES = {
    'Mumbai':    ['Andheri West', 'Andheri East', 'Goregaon East', 'Mulund West', 'Chembur', 'Powai', 'Mira Road', 'Borivali East', 'Malad West', 'Worli', 'Goregaon West', 'Malad East', 'Kandivali East', 'Borivali West', 'Kandivali West', 'Prabhadevi', 'Bandra West', 'Parel', 'Bhandup West', 'Santacruz West'],   # replace with your top 20
    'Bangalore': ['Whitefield', 'Sarjapur Road', 'Hebbal', 'Yelahanka', 'Marathahalli', 'Electronic City', 'Thanisandra', 'Bannerghatta Road', 'Devanahalli', 'Kanakapura Road', 'Kasavanahalli', 'Koramangala', 'HSR Layout', 'Varthur', 'Hennur Main Road', 'JP Nagar', 'KR Puram', 'Bagalur Main Road', 'Ramamurthy Nagar', 'Jakkur'],
    'New Delhi': ['Saket', 'Uttam Nagar', 'Chhattarpur', 'Dwarka Mor', 'Defence Colony', 'Laxmi Nagar', 'Janakpuri', 'Kalkaji', 'Pitampura', 'Paschim Vihar', 'Vikaspuri', 'Najafgarh', 'Vasant Kunj', 'Chittaranjan Park', 'Dwarka', 'St Nagar Burari', 'Sarita Vihar', 'Mayur Vihar 1', 'Patel Nagar West', 'Aya Nagar'],
    'Pune':      ['Hadapsar', 'Kharadi', 'Wagholi', 'Hinjewadi', 'Wakad', 'Baner', 'Lohegaon', 'Undri', 'Keshav Nagar', 'Dhanori', 'Manjri', 'Viman Nagar Central', 'Balewadi', 'Magarpatta City', 'Aundh', 'Wadgaon Sheri', 'Punawale', 'Pimpri Chinchwad', 'Kalyani Nagar', 'EON Free Zone'],
    'Nagpur':    ['Mihan', 'Manish Nagar', 'Besa', 'Wardha Road', 'Narendra Nagar', 'Hingna Road', 'Friends Colony', 'Manewada', 'Trimurti Nagar', 'Jamtha', 'Somalwada', 'Pipla', 'Kotewada', 'Bel Tarodi', 'Shivaji Nagar', 'Zingabai Takali', 'Mankapur', 'Pratap Nagar', 'Hudkeshwar', 'Wathoda']
}

FURNISHING = ['Furnished', 'Semi-Furnished', 'Unfurnished']  # replace with actual values
HOUSE_TYPE = ['Flat', 'House', 'Villa']     # replace with actual values

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Metro Rent Predictor",
    page_icon="🏙️",
    layout="centered"
)

st.title("🏙️ Metro City Rent Predictor")
st.markdown("Estimate monthly rent for residential properties across 5 Indian metro cities.")
st.divider()

# ── Input form ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", CITIES)
    locality = st.selectbox("Locality", LOCALITIES[city])
    house_type = st.selectbox("House Type", HOUSE_TYPE)
    furnishing = st.selectbox("Furnishing", FURNISHING)

with col2:
    area = st.number_input("Area (sq ft)", min_value=100.0, max_value=10000.0, value=1000, step=50)
    bhk = st.number_input("BHK", min_value=1, max_value=6, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2)
    balconies = st.number_input("Balconies", min_value=0, max_value=4, value=1)

st.divider()

# ── Prediction ───────────────────────────────────────────────
if st.button("Predict Rent", type="primary", use_container_width=True):
    
    # Build input dataframe — column names must exactly match training data
    input_data = pd.DataFrame([{
        'City': city,
        'locality': locality,
        'house_type': house_type,
        'furnishing': furnishing,
        'area': area,
        'bhk': bhk,
        'bathrooms': bathrooms,
        'balconies': balconies
    }])
    
    # Predict and reverse log transform
    pred_log = model.predict(input_data)[0]
    pred_rupees = np.expm1(pred_log)
    
    # Cap display at model's reliable range
    if pred_rupees > 180000:
        st.warning("⚠️ Prediction exceeds reliable range. Property may have unusual characteristics.")
    
    # Display result
    st.success(f"### Estimated Monthly Rent: ₹{pred_rupees:,.0f}")
    
    # Show confidence context
    if pred_rupees < 20000:
        st.info("📊 High confidence range — model performs best here.")
    elif pred_rupees < 50000:
        st.info("📊 Good confidence range.")
    else:
        st.info("📊 Moderate confidence — larger properties have higher prediction variance.")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Model: Gradient Boosting model  | R² = 0.78 | MAE = ₹13,575 | Trained on clipped data (≤₹1,80,000/month)")
st.caption("Cities: Mumbai · Bangalore · New Delhi · Pune · Nagpur")
