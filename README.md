# Malaria Detection Using Complete Blood Count (CBC)

An AI-powered machine learning application built with **Streamlit**, **Scikit-Learn**, and **Pandas** to predict malaria infection risk using Complete Blood Count (CBC) laboratory parameters.

---

## Features
- **Interactive Web Interface**: Clean and user-friendly UI built with Streamlit.
- **Risk Assessment**: Classifies whether a patient is likely positive or negative for Malaria based on CBC test values.
- **Machine Learning Pipeline**: Trained Random Forest model with preprocessing pipeline.
- **Key Parameters Analyzed**:
  - Age & Sex
  - Hemoglobin (Hb%)
  - Total WBC Count
  - Neutrophils & Lymphocytes (%)
  - Total Circulating Eosinophils
  - Hematocrit / Packed Cell Volume (HTC/PCV %)
  - MCH & MCHC
  - RDW-CV (%)
  - Platelet Count

---

## Project Structure
```
Malaria_CBC_Predictor/
├── data/
│   └── Malaria Diseases dataset - .csv   # Dataset for training
├── models/
│   └── malaria_model.pkl                 # Trained Random Forest pipeline
├── app.py                                # Streamlit web application
├── train_model.py                        # Model training and evaluation script
├── requirements.txt                      # Python dependencies
└── README.md                             # Documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhomendrakumar2-gif/malaria.git
   cd malaria
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Web App
```bash
streamlit run app.py
```

### Retrain the Model
```bash
python train_model.py
```

---

## Technologies Used
- **Python 3**
- **Streamlit**
- **Scikit-Learn**
- **Pandas & NumPy**
- **Joblib**
