"""
AI-Based Crop Recommendation System Using Soil Analysis Parameters
-------------------------------------------------------------------
File: app.py
Description: Interactive Streamlit web interface for real-time crop recommendation.
             Allows farmers or agricultural officers to input soil and weather
             parameters and view the predicted crop along with confidence scores.

Author: Senior Machine Learning Engineer (Antigravity)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "crop_model.pkl"
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Crop icons mapping for visual appeal
CROP_ICONS = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🧆", "kidneybeans": "🫘",
    "pigeonpeas": "🌱", "mothbeans": "🌱", "mungbean": "🌱", "blackgram": "🌱",
    "lentil": "🍲", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍏",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "🧶",
    "jute": "🧵", "coffee": "☕"
}

@st.cache_resource
def load_trained_model():
    """Load the serialized Random Forest model from disk with caching."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def main():
    # Header Section
    st.title("🌱 AI-Based Crop Recommendation System")
    st.markdown(
        "**Precision Agriculture Decision Support System**: Input soil nutrient levels and "
        "environmental parameters to receive data-driven crop recommendations powered by Machine Learning."
    )
    st.markdown("---")

    # Load Model
    model = load_trained_model()
    if model is None:
        st.error(
            f"⚠️ Model file `{MODEL_PATH}` not found! "
            "Please run `python train.py` first to train and generate the model."
        )
        return

    # Sidebar for presets and quick guide
    with st.sidebar:
        st.header("⚙️ Demo Presets")
        st.caption("Click a preset to quickly populate sample values for demonstration:")

        preset = st.radio(
            "Select Preset Profile:",
            ("Custom Input", "🌾 Rice (High N, High Rain)", "🍏 Apple (High P & K, Cool)", "☕ Coffee (Moderate NPK, Humid)")
        )

        st.markdown("---")
        st.header("ℹ️ Parameter Guide")
        st.markdown("""
        - **N (Nitrogen)**: Essential for vegetative growth and green foliage.
        - **P (Phosphorus)**: Critical for root architecture and flowering.
        - **K (Potassium)**: Regulates water balance and disease resistance.
        - **Temperature**: Ambient air temperature (°C).
        - **Humidity**: Relative atmospheric humidity (%).
        - **pH**: Soil acidity/alkalinity (Optimal: 6.0 - 7.5).
        - **Rainfall**: Seasonal rainfall level (mm).
        """)

    # Default values based on preset selection
    if preset == "🌾 Rice (High N, High Rain)":
        def_n, def_p, def_k = 90, 42, 43
        def_temp, def_hum, def_ph, def_rain = 20.8, 82.0, 6.5, 202.9
    elif preset == "🍏 Apple (High P & K, Cool)":
        def_n, def_p, def_k = 20, 134, 198
        def_temp, def_hum, def_ph, def_rain = 22.5, 92.0, 5.9, 110.0
    elif preset == "☕ Coffee (Moderate NPK, Humid)":
        def_n, def_p, def_k = 101, 30, 30
        def_temp, def_hum, def_ph, def_rain = 25.5, 60.0, 6.8, 160.0
    else:  # Custom / dataset median
        def_n, def_p, def_k = 40, 50, 45
        def_temp, def_hum, def_ph, def_rain = 25.0, 70.0, 6.5, 100.0

    # Layout: Two input columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Soil Nutrient Parameters")
        n_val = st.slider("Nitrogen (N) in soil (kg/ha)", min_value=0, max_value=145, value=int(def_n), step=1)
        p_val = st.slider("Phosphorus (P) in soil (kg/ha)", min_value=5, max_value=145, value=int(def_p), step=1)
        k_val = st.slider("Potassium (K) in soil (kg/ha)", min_value=5, max_value=205, value=int(def_k), step=1)
        ph_val = st.slider("Soil pH Level (0 - 14)", min_value=3.5, max_value=10.0, value=float(def_ph), step=0.1)

    with col2:
        st.subheader("🌤️ Environmental Parameters")
        temp_val = st.slider("Temperature (°C)", min_value=8.0, max_value=45.0, value=float(def_temp), step=0.5)
        hum_val = st.slider("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=float(def_hum), step=1.0)
        rain_val = st.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=float(def_rain), step=1.0)

    st.markdown("")
    predict_btn = st.button("🔍 Recommend Optimal Crop", type="primary", use_container_width=True)

    if predict_btn:
        # Build DataFrame matching feature names
        input_df = pd.DataFrame([[
            float(n_val), float(p_val), float(k_val),
            float(temp_val), float(hum_val), float(ph_val), float(rain_val)
        ]], columns=FEATURE_NAMES)

        # Inference
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_

        top3_indices = np.argsort(probs)[::-1][:3]
        top_crop = prediction
        top_conf = probs[top3_indices[0]] * 100
        crop_icon = CROP_ICONS.get(top_crop.lower(), "🌱")

        st.markdown("---")
        st.subheader("🎯 Recommendation Result")

        # Main recommendation card
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.success(f"### {crop_icon} {top_crop.upper()}")
            st.metric(label="Model Confidence", value=f"{top_conf:.2f}%")
            st.caption("Primary recommendation based on optimal growth suitability.")

        with res_col2:
            st.write("#### 📊 Top 3 Candidate Crops:")
            top_data = []
            for rank, idx in enumerate(top3_indices, start=1):
                c_name = classes[idx]
                c_conf = probs[idx] * 100
                c_icon = CROP_ICONS.get(c_name.lower(), "🌱")
                top_data.append({
                    "Rank": f"#{rank}",
                    "Crop": f"{c_icon} {c_name.capitalize()}",
                    "Confidence (%)": round(c_conf, 2)
                })

            top_df = pd.DataFrame(top_data)
            st.dataframe(top_df, hide_index=True, use_container_width=True)
            st.bar_chart(top_df.set_index("Crop")["Confidence (%)"])

    # Tabs for Model Details & Project Info
    st.markdown("---")
    tab1, tab2 = st.tabs(["📈 Model Performance & Comparison", "📖 System Architecture & Details"])

    with tab1:
        st.write("### Model Evaluation on Unseen Test Data (80:20 Split)")
        perf_data = {
            "Model": ["Random Forest (Selected)", "Decision Tree", "K-Nearest Neighbors (Scaled)"],
            "Accuracy": ["99.55%", "97.95%", "97.95%"],
            "Precision (Weighted)": ["99.57%", "98.06%", "98.04%"],
            "Recall (Weighted)": ["99.55%", "97.95%", "97.95%"],
            "F1-Score (Weighted)": ["99.55%", "97.94%", "97.93%"]
        }
        st.table(pd.DataFrame(perf_data))
        st.info("💡 **Why Random Forest is the best:** It aggregates predictions across 100 randomized decision trees (bagging), reducing individual tree variance and effectively mitigating overfitting.")

    with tab2:
        st.write("### Project Architecture")
        st.markdown("""
        1. **Data Preprocessing**: Features (N, P, K, temperature, humidity, pH, rainfall) and target crop labels separated.
        2. **Stratified Split**: 80% training (1,760 samples) and 20% testing (440 samples) maintaining exact class balance (80/20 per crop).
        3. **Data Leakage Prevention**: Preprocessing scalers are strictly fitted on training splits only.
        4. **Inference**: Random Forest ensemble outputs probability distribution over all 22 crop classes using `predict_proba`.
        """)


if __name__ == "__main__":
    main()

