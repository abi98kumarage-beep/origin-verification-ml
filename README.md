# Origin Verification ML System

## Project Summary

An end-to-end machine learning system that predicts crop origin using environmental and soil features. Inspired by real-world supply chain verification problems, this project demonstrates statistical modelling, machine learning, and uncertainty-aware predictions.

---

## Key Highlights

* Built **end-to-end ML pipeline** from raw data to deployment
* Combined **statistical (Logistic Regression)** and **ML models (Random Forest)**
* Achieved **~99% accuracy** on multi-class classification
* Implemented **uncertainty estimation using prediction confidence**
* Deployed an **interactive dashboard using Streamlit**

---

## Business Context

This project simulates how organisations verify product origin using data. It aligns with real-world challenges such as:

* supply chain transparency
* fraud detection
* origin authentication

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## How to Run

### 1. Clone repo

```bash
git clone https://github.com/abi98kumarage-beep/origin-verification-ml.git
cd origin-verification-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train model

```bash
python run_pipeline.py
```

### 4. Run dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Features

* Multi-class classification (22 crop types)
* Model comparison (Logistic vs Random Forest)
* Confidence-based predictions
* Interactive dashboard

---

##  Project Structure

```
origin-verification/
│── data/
│── models/
│── app/
│── run_pipeline.py
│── requirements.txt
│── README.md
```

---

##  Model Performance

* Logistic Regression: ~97% accuracy
* Random Forest: ~99% accuracy

---

##  Example Output

* Predicted Crop: Rice
* Confidence: 0.99

---

##  Future Improvements

* Add SHAP explainability
* Deploy API using FastAPI
* Add cloud deployment
* Improve uncertainty modelling

---

## Author

**Abimani Kumarage**
Data Analytics & Machine Learning
