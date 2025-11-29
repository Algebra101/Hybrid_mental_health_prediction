import streamlit as st
import joblib
from recommender import get_recommendations

# Load Model (Uncomment when you have trained it)
# model = joblib.load('../models/hybrid_model.pkl')

st.set_page_config(page_title="Mental Health Predictor", layout="wide")

st.title("🧠 Hybrid Mental Health Prediction System")
st.markdown("### A Multi-Modal Approach combining Behavioral Metrics & Journaling")

# --- COL 1: Structured Input (Behavioral) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Behavioral Profile")
    sleep = st.slider("Average Sleep (Hours)", 0, 12, 6)
    cgpa = st.number_input("Current CGPA", 0.0, 4.0, 3.0)
    stress = st.selectbox("Do you feel Academic Pressure?",)
    anxiety = st.selectbox("Do you have Anxiety?",)

# --- COL 2: Unstructured Input (NLP) ---
with col2:
    st.subheader("Step 2: Mental Journal")
    journal_text = st.text_area("How are you feeling today? (Be honest)", height=200)

# --- Prediction Button ---
if st.button("Analyze Mental Health Risk"):
    if journal_text:
        # 1. Preprocess Inputs
        # 2. Make Prediction using loaded model
        # prediction = model.predict(...) 
        
        # Placeholder for demo until model is trained
        prediction = "Moderate Risk" 
        
        st.divider()
        st.markdown(f"### Predicted Status: **{prediction}**")
        
        # --- Recommendation Engine Output ---
        st.subheader("Recommended Interventions:")
        recommendations = get_recommendations(prediction)
        for rec in recommendations:
            st.info(f"• {rec}")
    else:
        st.warning("Please fill in the journal entry.")