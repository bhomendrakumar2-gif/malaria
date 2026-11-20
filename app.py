
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="MalariaDetect AI",
    page_icon="🩸",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.main-title {
    text-align: center;
    color: #12345a;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    color: #12345a;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}

div.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    background-color: #1565c0;
    color: white;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background-color: #0d47a1;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# HEADER
# ----------------------------------

st.markdown(
    '<div class="main-title">🩸 MalariaDetect AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">CBC Report Based Malaria Research Tool</div>',
    unsafe_allow_html=True
)

st.error(
    "Educational research prototype only. CBC values cannot "
    "confirm or rule out malaria. A laboratory microscopy or "
    "Rapid Diagnostic Test (RDT) is required for diagnosis."
)

# ----------------------------------
# LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = Path("models/malaria_model.pkl")

if not MODEL_PATH.exists():
    st.warning("Trained model not found. Please train the model first.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ----------------------------------
# PATIENT INFORMATION FORM
# ----------------------------------

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (Years)",
            min_value=1,
            max_value=120,
            value=None,
            placeholder="Enter patient age"
        )

    with col2:
        sex = st.selectbox(
            "Gender",
            ["Select Gender", "Male", "Female"]
        )

    st.divider()

    st.markdown(
        '<div class="section-title">🧪 CBC Blood Report Values</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        hemoglobin = st.number_input(
            "Hemoglobin (g/dL)",
            min_value=0.0,
            value=None,
            placeholder="Enter hemoglobin"
        )

        wbc = st.number_input(
            "Total WBC Count (/cumm)",
            min_value=0.0,
            value=None,
            placeholder="Enter WBC count"
        )

        neutrophils = st.number_input(
            "Neutrophils (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            placeholder="Enter neutrophils"
        )

        lymphocytes = st.number_input(
            "Lymphocytes (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            placeholder="Enter lymphocytes"
        )

        eosinophils = st.number_input(
            "Total Circulating Eosinophils",
            min_value=0.0,
            value=None,
            placeholder="Enter eosinophils"
        )

    with col2:
        hct = st.number_input(
            "HTC/PCV (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            placeholder="Enter HCT"
        )

        mch = st.number_input(
            "MCH (pg)",
            min_value=0.0,
            value=None,
            placeholder="Enter MCH"
        )

        mchc = st.number_input(
            "MCHC (g/dL)",
            min_value=0.0,
            value=None,
            placeholder="Enter MCHC"
        )

        rdw = st.number_input(
            "RDW-CV (%)",
            min_value=0.0,
            value=None,
            placeholder="Enter RDW"
        )

        platelets = st.number_input(
            "Platelet Count (/cumm)",
            min_value=0.0,
            value=None,
            placeholder="Enter platelet count"
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Analyze CBC Report",
        type="primary"
    )

# ----------------------------------
# PREDICTION
# ----------------------------------

if submitted:

    values = [
        age,
        hemoglobin,
        wbc,
        neutrophils,
        lymphocytes,
        eosinophils,
        hct,
        mch,
        mchc,
        rdw,
        platelets
    ]

    if any(value is None for value in values):
        st.warning("Please fill in all CBC values before analysis.")

    elif sex == "Select Gender":
        st.warning("Please select patient gender.")

    else:

        # Create input using exact training dataset columns
        patient = pd.DataFrame([{
            "Age": age,
            "Hemoglobin(Hb%)": hemoglobin,
            "Total WBC count(/cumm)": wbc,
            "Neutrophils": neutrophils,
            "Lymphocytes": lymphocytes,
            "Total Cir.Eosinophils": eosinophils,
            "HTC/PCV(%)": hct,
            "MCH(pg)": mch,
            "MCHC(g/dl)": mchc,
            "RDW-CV(%)": rdw,
            "Platelet Count": platelets,
            "Sex": sex
        }])

        try:

            prediction = model.predict(patient)[0]

            probabilities = model.predict_proba(patient)[0]

            classes = list(model.classes_)

            positive_index = classes.index("positive")

            score = probabilities[positive_index]

            st.divider()

            st.subheader("📊 AI Model Output")

            st.metric(
                label="Positive-class model score",
                value=f"{score * 100:.2f}%"
            )

            st.write(
                "Model classification:",
                prediction.capitalize()
            )

            st.info(
                "This is an experimental machine learning output, "
                "not the probability that the patient has malaria. "
                "CBC patterns overlap across many illnesses. "
                "Please obtain a laboratory malaria test."
            )

            with st.expander("View Entered Patient Information"):
                st.dataframe(patient, use_container_width=True)

        except Exception as e:

            st.error(f"Prediction error: {e}")

# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.caption(
    "MalariaDetect AI | Educational Machine Learning Project | "
    "Not intended for medical diagnosis."
)
