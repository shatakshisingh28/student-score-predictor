# 🎓 Student Exam Score Predictor

A beginner-friendly Machine Learning project that predicts a student's exam score based on key study habits and performance metrics.

---

## 📌 What This Project Does

Given 5 inputs about a student, the model predicts their expected exam score (0–100):

| Feature | Description |
|---|---|
| `study_hours` | Hours studied per day |
| `sleep_hours` | Hours slept per night |
| `attendance_pct` | Class attendance percentage |
| `prev_score` | Score in previous exam |
| `extra_classes` | Attends extra coaching (yes/no) |

---

## 🗂️ Project Structure

```
student-score-predictor/
│
├── app/
│   └── index.html          ← Interactive web UI (open in browser)
│
├── model/
│   ├── train_model.py       ← Train the ML model
│   ├── predict.py           ← Run predictions from Python
│   ├── model.pkl            ← Saved trained model (generated)
│   └── scaler.pkl           ← Saved scaler (generated)
│
├── data/
│   └── student_data.csv     ← Generated dataset (500 rows)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
cd model
python train_model.py
```

### 3. Make a prediction (Python)
```bash
python predict.py
```

### 4. Use the Web App
Just open `app/index.html` in any browser — no server needed!

---

## 🧠 ML Concepts Used

- **Linear Regression** — learns a weighted equation for the target
- **Train/Test Split** — 80% training, 20% testing
- **StandardScaler** — normalizes features for better accuracy
- **MAE & R² Score** — metrics to evaluate model performance

---

## 📊 Expected Model Performance

| Metric | Value |
|---|---|
| Mean Absolute Error | ~3–5 points |
| R² Score | ~0.95+ |

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
```

---

## 💡 Ideas to Extend This Project

- Add more features (e.g. internet usage, parental education)
- Try other models: Random Forest, Ridge Regression
- Build a Flask/FastAPI backend to serve real predictions
- Visualize feature importance with matplotlib
- Deploy to Hugging Face Spaces or Streamlit Cloud

---

## 🧑‍💻 Author

Built as a beginner data science portfolio project using Python and scikit-learn.
