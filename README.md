# 💳 Customer Retention Intelligence System

An end-to-end Machine Learning system to predict customer churn using FastAPI and Streamlit with optimized threshold tuning.

---

## 🚀 Live Demo
👉 (Add after deployment)

---

## 📊 Key Features

- Predicts customer churn probability
- Uses optimized decision threshold (not default 0.5)
- FastAPI backend for real-time inference
- Streamlit dashboard with risk insights
- Business-driven decision logic

---

## 🧠 ML Highlights

- ROC-AUC: ~0.86
- Recall improved from 44% → 75%
- Threshold optimization based on business constraints

---

## 🏗️ Tech Stack

- Python
- Scikit-learn
- FastAPI
- Streamlit
- Pandas / NumPy

---

## ⚙️ How it Works

1. User inputs customer data
2. API processes data through ML pipeline
3. Model outputs probability
4. Optimized threshold applied
5. UI displays risk + recommendations

---

## ▶️ Run Locally

```bash
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
