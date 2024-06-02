import pandas as pd
from sklearn.datasets import load_diabetes
import streamlit as st

def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat([X, Y], axis=1)
        st.markdown('The **Diabetes** dataset is used as the example.')
    return df
