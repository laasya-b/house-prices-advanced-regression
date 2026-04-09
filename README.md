# 🏠 House Prices - Advanced Regression Techniques

A machine learning web application that predicts residential house prices based on key property features. Built as part of the [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), this project demonstrates an end-to-end ML pipeline from raw data to a deployed interactive web app.

🔗 **[Live App →](https://house-prices-advanced-regression-lzca4pe9burpr3ybnzl3w3.streamlit.app/)**

---

## 📌 Project Overview

This project walks through the full data science workflow:
- Exploratory Data Analysis (EDA)
- Data cleaning & feature engineering
- Model training & evaluation
- Interactive web app deployment

The final model achieves an **R² score of 0.88** and an **RMSE of ~$29,718** on the test set.

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (Random Forest) |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git & GitHub |

---

## 📊 Model Performance

| Model | RMSE | R² Score |
|---|---|---|
| Linear Regression | $34,281 | 0.847 |
| **Random Forest** | **$29,718** | **0.885** |

Random Forest was selected as the final model based on superior performance across both metrics.

---

## 🔍 Key Features of the App

- **Interactive sliders** to input house features
- **Instant price prediction** powered by a trained Random Forest model
- **Feature importance chart** showing which factors drive house prices most
- **Clean, responsive UI** built with Streamlit

---

## 📂 Dataset

- **Source:** [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Size:** 1,460 training samples, 81 features
- **Target:** `SalePrice` — the property sale price in USD

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/laasya-b/house-prices-advanced-regression.git
cd house-prices-advanced-regression

# Install dependencies
pip install -r requirements.txt

# Run the app
cd app
streamlit run app.py
```

---


*Built with Python, Scikit-learn & Streamlit*

