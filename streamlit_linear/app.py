import streamlit as st
import numpy as np
import joblib
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Coffee Sales Prediction",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Coffee Sales Prediction App")
st.write("Predict coffee sales using Machine Learning")

# --------------------------------------------------
# Load Model, Scaler, Feature Selector (Safe Paths)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    selector_path = os.path.join(BASE_DIR, "feature_selector.pkl")

    if not os.path.exists(model_path):
        st.error("❌ model.pkl not found in repository")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error("❌ scaler.pkl not found in repository")
        st.stop()
    if not os.path.exists(selector_path):
        st.error("❌ feature_selector.pkl not found in repository")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    return model, scaler, selector

model, scaler, selector = load_artifacts()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("📥 Enter Input Features")

col1, col2 = st.columns(2)

with col1:
    Temperature_C = st.number_input("Temperature (°C)", value=25.0)
    Is_Weekend = st.selectbox("Is Weekend?", [0, 1])
    Is_Raining = st.selectbox("Is Raining?", [0, 1])

with col2:
    Num_Customers = st.number_input("Number of Customers", value=100, step=1)
    Staff_Count = st.number_input("Staff Count", value=5, step=1)
    Promotion_Active = st.selectbox("Promotion Active?", [0, 1])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🚀 Predict Sales"):
    try:
        input_data = np.array([[
            Temperature_C,
            Is_Weekend,
            Is_Raining,
            Num_Customers,
            Staff_Count,
            Promotion_Active
        ]])

        # Debug info (helps if something breaks)
        st.write("Input shape:", input_data.shape)

        # Step 1: Scaling
        scaled_data = scaler.transform(input_data)

        # Step 2: Feature Selection
        selected_data = selector.transform(scaled_data)

        # Step 3: Prediction
        prediction = model.predict(selected_data)

        st.success(f"📈 Predicted Coffee Sales: **{prediction[0]:.2f} units**")

    except Exception as e:
        st.error("❌ Something went wrong while predicting.")
        st.exception(e)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
