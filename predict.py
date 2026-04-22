# ============================================================
#  Student Exam Score Predictor
#  Prediction Utility
# ============================================================

import pickle
import numpy as np

def load_artifacts():
    """Load trained model and scaler from disk."""
    with open("model/model.pkl",  "rb") as f: model  = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f: scaler = pickle.load(f)
    return model, scaler


def predict_score(study_hours, sleep_hours, attendance_pct,
                  prev_score, extra_classes):
    """
    Predict exam score for a single student.

    Parameters
    ----------
    study_hours     : float  – hours studied per day        (1–10)
    sleep_hours     : float  – hours slept per night        (4–9)
    attendance_pct  : float  – attendance percentage        (50–100)
    prev_score      : float  – score in previous exam       (0–100)
    extra_classes   : int    – attends extra coaching? 0/1

    Returns
    -------
    float  – predicted exam score (0–100)
    """
    model, scaler = load_artifacts()

    features = np.array([[
        study_hours, sleep_hours, attendance_pct,
        prev_score, extra_classes
    ]])

    features_scaled = scaler.transform(features)
    score = model.predict(features_scaled)[0]
    return float(np.clip(score, 0, 100))


def get_feedback(score):
    """Return a human-readable grade band and advice."""
    if score >= 85:
        return "Excellent 🏆", "Outstanding performance! Keep up the great work."
    elif score >= 70:
        return "Good 👍",      "Solid result. A little more focus and you'll hit excellence."
    elif score >= 55:
        return "Average 📘",   "You're on track. Try increasing your study hours."
    elif score >= 40:
        return "Below Average ⚠️", "Consider attending extra classes and improving sleep."
    else:
        return "At Risk 🚨",   "Urgent: seek academic support and revise your study plan."


# Quick CLI test
if __name__ == "__main__":
    score = predict_score(
        study_hours=6,
        sleep_hours=7,
        attendance_pct=85,
        prev_score=72,
        extra_classes=1
    )
    grade, advice = get_feedback(score)
    print(f"\nPredicted Score : {score:.1f} / 100")
    print(f"Grade Band      : {grade}")
    print(f"Advice          : {advice}")
