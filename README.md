# 💎 Real-Time Smart Loan Eligibility Advisor
### Production-Grade AI + Explainable Intelligence Platform

## 🧠 5 ML Models + Ensemble
| Algorithm | Accuracy | AUC-ROC |
|-----------|----------|---------|
| Logistic Regression | 93.33% | 0.9855 |
| Decision Tree | 89.33% | 0.9551 |
| Random Forest | 91.33% | 0.9803 |
| Gradient Boosting | 91.67% | 0.9799 |
| Neural Network (ANN) | 93.00% | 0.9831 |
| **Ensemble (Weighted)** | **92.33%** | **0.9788** |

## 🚀 Quick Start
```bash
pip install flask flask-cors scikit-learn pandas numpy joblib
python dataset_generate.py && python model_train.py && python backend_app.py
# Frontend: drop SmartLoanAdvisor.jsx into React project, npm install recharts
```

## 📁 Files
- SmartLoanAdvisor.jsx — Full React frontend (5 pages + chatbot + simulator)
- backend_app.py — Flask API (/predict /simulate /analytics)
- model_train.py — Trains all models, saves .pkl files
- dataset_generate.py — Generates 3000-sample synthetic dataset
- loan_data.csv — Training dataset

## Features
- Real-time ensemble prediction (5 models + weighted voting)
- SVG risk gauge meter (speedometer)
- What-If simulator with 8 scenario comparisons
- SHAP-style feature importance per prediction
- Plain-English rejection explanations + improvement tips
- ROC curves, confusion matrix, radar chart analytics
- AI chatbot with quick-suggestion buttons
