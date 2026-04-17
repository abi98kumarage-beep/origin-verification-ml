import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/raw/Crop_recommendation.csv")

print("\n✅ Data Loaded")
print(df.head())


# =========================
# 2. SPLIT DATA
# =========================
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 3. SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 4. LOGISTIC REGRESSION
# =========================
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train)

log_preds = log_model.predict(X_test_scaled)

print("\n📊 Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))


# =========================
# 5. RANDOM FOREST
# =========================
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("\n🌲 Random Forest Accuracy:", accuracy_score(y_test, rf_preds))


# =========================
# 6. REPORT
# =========================
print("\n📄 Classification Report:\n")
print(classification_report(y_test, rf_preds))

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# SAFETY CHECK
if rf_model is None:
    raise Exception("Model training failed - rf_model is None")

# Save model
joblib.dump(rf_model, "models/model.pkl")

print("\n💾 Model saved successfully")

# Verify file size
size = os.path.getsize("models/model.pkl")
print("📦 Model size (bytes):", size)

