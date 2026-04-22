# ============================================================
#  Student Exam Score Predictor
#  Model Training Script
#  Author: [Your Name]
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ------------------------------------------------------------
# 1. Generate synthetic training data
# ------------------------------------------------------------

np.random.seed(42)
n_samples = 500

def generate_dataset(n):
    """
    Creates a realistic synthetic dataset of student performance.
    Features:
      - study_hours     : hours studied per day (1–10)
      - sleep_hours     : hours slept per night (4–9)
      - attendance_pct  : class attendance percentage (50–100)
      - prev_score      : score in previous exam (40–100)
      - extra_classes   : whether student attends extra classes (0/1)
    Target:
      - exam_score      : final exam score (0–100)
    """
    study_hours    = np.random.uniform(1, 10, n)
    sleep_hours    = np.random.uniform(4, 9, n)
    attendance_pct = np.random.uniform(50, 100, n)
    prev_score     = np.random.uniform(40, 100, n)
    extra_classes  = np.random.randint(0, 2, n)

    # Score formula: weighted combination + noise
    exam_score = (
        3.5 * study_hours
        + 1.2 * sleep_hours
        + 0.3 * attendance_pct
        + 0.25 * prev_score
        + 4.0 * extra_classes
        + np.random.normal(0, 3, n)  # realistic noise
    )

    # Clamp scores between 0 and 100
    exam_score = np.clip(exam_score, 0, 100)

    return pd.DataFrame({
        "study_hours":    study_hours,
        "sleep_hours":    sleep_hours,
        "attendance_pct": attendance_pct,
        "prev_score":     prev_score,
        "extra_classes":  extra_classes,
        "exam_score":     exam_score
    })

df = generate_dataset(n_samples)

# Save dataset for reference
os.makedirs("data", exist_ok=True)
df.to_csv("data/student_data.csv", index=False)
print(f"✔ Dataset saved  →  data/student_data.csv  ({len(df)} rows)")

# ------------------------------------------------------------
# 2. Prepare features and target
# ------------------------------------------------------------

FEATURES = ["study_hours", "sleep_hours", "attendance_pct", "prev_score", "extra_classes"]
TARGET   = "exam_score"

X = df[FEATURES]
y = df[TARGET]

# Train / test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ------------------------------------------------------------
# 3. Train Linear Regression model
# ------------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Evaluate
# ------------------------------------------------------------

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n📊 Model Evaluation")
print(f"   Mean Absolute Error : {mae:.2f} points")
print(f"   R² Score            : {r2:.4f}")
print(f"\n📈 Feature Coefficients")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"   {feat:<20} {coef:+.4f}")

# ------------------------------------------------------------
# 5. Save model & scaler
# ------------------------------------------------------------

os.makedirs("model", exist_ok=True)
with open("model/model.pkl",  "wb") as f: pickle.dump(model,  f)
with open("model/scaler.pkl", "wb") as f: pickle.dump(scaler, f)

print("\n✔ Model saved  →  model/model.pkl")
print("✔ Scaler saved →  model/scaler.pkl")
