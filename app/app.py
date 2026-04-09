import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'model.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))

#  Page Config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Predictor")
st.markdown("Predict the sale price of a house based on its features using a Random Forest ML model.")
st.divider()


st.sidebar.header("🔧 Enter House Features")

overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500)
total_sf = st.sidebar.number_input("Total Square Footage", 500, 10000, 2000)
year_built = st.sidebar.number_input("Year Built", 1872, 2024, 2000)
garage_cars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
total_bathrooms = st.sidebar.slider("Total Bathrooms", 0, 6, 2)
house_age = st.sidebar.number_input("House Age (years)", 0, 150, 20)
total_bsmt_sf = st.sidebar.number_input("Basement Area (sq ft)", 0, 3000, 800)
fireplaces = st.sidebar.slider("Number of Fireplaces", 0, 3, 1)
neighborhood = st.sidebar.slider("Neighborhood Score (encoded 0-24)", 0, 24, 12)


input_data = pd.DataFrame(
    [np.zeros(len(feature_names))],
    columns=feature_names
)


user_inputs = {
    'OverallQual': overall_qual,
    'GrLivArea': gr_liv_area,
    'TotalSF': total_sf,
    'YearBuilt': year_built,
    'GarageCars': garage_cars,
    'TotalBathrooms': total_bathrooms,
    'HouseAge': house_age,
    'TotalBsmtSF': total_bsmt_sf,
    'Fireplaces': fireplaces,
    'Neighborhood': neighborhood
}

for feature, value in user_inputs.items():
    if feature in input_data.columns:
        input_data[feature] = value


col1, col2, col3 = st.columns(3)

with col2:
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)

if predict_btn:
    prediction = model.predict(input_data)[0]

    st.divider()
    st.markdown("### 🎯 Predicted Sale Price")

    col1, col2, col3 = st.columns(3)
    with col2:
        st.metric(
            label="Estimated Price",
            value=f"${prediction:,.0f}"
        )

   
    st.divider()
    st.markdown("### 📊 Top 15 Most Important Features")

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)

    fig = px.bar(
        feat_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='blues',
        title='Feature Importance (Random Forest)'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built with Python, Scikit-learn & Streamlit | Dataset: Kaggle House Prices Competition")