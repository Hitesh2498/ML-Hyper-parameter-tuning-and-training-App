import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_data
from params import get_params
from model import build_model

st.set_page_config(page_title='The Machine Learning Hyperparameter Optimization App', layout='wide')

st.write("""
# The Machine Learning Hyperparameter Optimization App
**(Regression Edition)**

In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
""")

# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Get parameters from params.py
split_size, param_grid, rf_params = get_params()

# Main panel
st.subheader('Dataset')

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write(df)
    build_model(df, split_size, param_grid, rf_params)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        df = load_data()
        st.write(df.head(5))
        build_model(df, split_size, param_grid, rf_params)
