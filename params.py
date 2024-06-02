import streamlit as st
import numpy as np

def get_params():
    st.sidebar.header('Set Parameters')
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.sidebar.subheader('Learning Parameters')
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10,50), 50)
    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
    st.sidebar.write('---')
    parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1,3), 1)
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.sidebar.subheader('General Parameters')
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['squared_error', 'poisson', 'absolute_error', 'friedman_mse'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
    max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+1, 1)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

    rf_params = {
        'n_estimators': parameter_n_estimators,
        'random_state': parameter_random_state,
        'max_features': parameter_max_features,
        'criterion': parameter_criterion,
        'min_samples_split': parameter_min_samples_split,
        'min_samples_leaf': parameter_min_samples_leaf,
        'bootstrap': parameter_bootstrap,
        'oob_score': parameter_oob_score,
        'n_jobs': parameter_n_jobs
    }

    return split_size, param_grid, rf_params
