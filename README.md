# The Machine Learning Hyperparameter Optimization App

The Machine Learning Hyperparameter Optimization App is a Streamlit-based web application designed for optimizing hyperparameters of a RandomForestRegressor model for regression tasks. The app allows users to upload their own dataset, set hyperparameters through an intuitive interface, and visualize the results of the optimization process.

## Features :-

1. **CSV File Upload:** Users can upload their own dataset in CSV format.
2. **Parameter Configuration:** Interactive sliders and inputs for setting hyperparameters.
3. **Model Building:** Uses RandomForestRegressor to build regression models.
4. **Grid Search Optimization:** Performs hyperparameter optimization using GridSearchCV.
5. **Performance Metrics:** Displays model performance metrics including R² score.
6. **Visualization:** 3D surface plot to visualize the results of hyperparameter tuning.
7. **Example Dataset:** Provides an example dataset (Diabetes dataset) for demonstration purposes.
8. **CSV Download:** Allows downloading the grid search results as a CSV file.

## Installation :- 
**1. Clone the repository:**
   
   ```python
   git clone https://github.com/yourusername/ml-hyperparameter-optimization-app.git
   ```
**2. Install required packages:**

```python
pip install -r requirements.txt
```
**3. Run the App:**
```python
streamlit run app.py
```
