# 🏙️ Metro City Rent Predictor

## 🚀 Live Demo
[Try the app](https://rent-prediction-ib6uec4lrm9mv8gfeug9uo.streamlit.app)

---

## 📌 Project Overview
An end-to-end machine learning project that predicts monthly residential 
rent across 5 Indian metro cities. Built with Python and scikit-learn, 
deployed via Streamlit Cloud.

---

## 🏙️ Cities Covered
| City | Training Samples | Reliability |
|------|-----------------|-------------|
| New Delhi | 1,753 | High |
| Bangalore | 1,698 | High |
| Pune | 1,756 | High |
| Mumbai | 1,530 | High |
| Nagpur | 594 | Moderate |

---

## 📊 Dataset
- **Source:** Kaggle — India House Rent Prediction
- **Size:** 7,537 rows after cleaning
- **Target:** Monthly rent (log-transformed with log1p)
- **Rent range:** ₹5,000 – ₹1,80,000/month
  (properties above ₹1,80,000 capped — dataset contained
  extreme outliers up to ₹27,00,000 likely from data entry errors)

### Features used:
- City, Locality, House Type, Furnishing Status
- Area (sq ft), BHK, Bathrooms, Balconies

---

## 🤖 Model Iteration History

| Version | R² Score | MAE | Notes |
|---------|----------|-----|-------|
| v1 | 0.990 | — | Inflated — data leakage via area_rate feature |
| v2 | 0.782 | ₹17,339 | Clean baseline — leakage removed |
| v3 | 0.78 | ₹13,581 | Random Forest on clipped + filtered data |

### Why v1 scored 0.99 — and why that was wrong
The original model included `area_rate` (rent ÷ area) as a feature —
a direct mathematical derivative of the target variable. This caused
the model to recover rent algebraically rather than learn genuine
market patterns. After removing this feature, R² reflects true
predictive performance.

---

## ⚙️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Pipeline, GradientBoostingRegressor, RandomizedSearchCV)
- Matplotlib / Seaborn
- Streamlit
- Jupyter Notebook

---

## 📈 Model Performance
- **Algorithm:** Random Forest Regressor
- **R² Score:** 0.78
- **MAE:** ₹13,581
- **Validation:** Train/test split (80/20, random_state=42)
- **Target transform:** log1p applied to Rent

---

## ⚠️ Known Limitations
- Covers 5 cities only — not a pan-India predictor
- Predictions less reliable for Nagpur (fewer training samples)
- Static dataset — does not reflect real-time market prices
- Higher error for properties above ₹1,00,000/month

---

## 🗺️ Roadmap
- [ ] Add metro distance feature (geopy)
- [ ] Per-city MAE breakdown
- [ ] Periodic retraining with fresh data

---

## 👤 Author
**Zaid Bin Altaf**

If you find this project useful, feel free to ⭐ star the repository.
