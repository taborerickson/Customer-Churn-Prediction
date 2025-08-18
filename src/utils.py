# src/utils.py 
# Utility functions 

from pathlib import Path 
import os 
import pandas as pd 

# Defining project paths 
base_path = Path(__file__).resolve.parents[1] 
data_path = base_path / "Data" 
model_path = base_path / "Models" 
visuals_path = base_path / "Visuals" 

# Making sure folders exists 
for path in [data_path, model_path, visuals_path]: 
    path.mkdir(parents=True, exist_ok=True) 

# Function to save data files to Data/ folder 
def save_data(data: pd.DataFrame, file_name: str): 
    file_path = os.path.join(data_path, f"{file_name}.csv") 
    data.to_csv(file_path, index=False) 
    print(f"Saved -> {file_path}") 

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np 
# Helper function to calculate cross-tabulations and group churn rates 
def group_churn(data, group_column, target='Churn'): 
    # In-group churn 
    cross_tab = pd.crosstab(data[group_column], data[target], normalize='index') 
    cross_tab = cross_tab.rename(columns={'No': 'churn_no_pct', 'Yes': 'churn_yes_pct'}) 
    # Population comparison 
    population_share = data[group_column].value_counts(normalize=True).rename('population_share') 
    result = cross_tab.join(population_share) 
    result['group_churn'] = result['churn_yes_pct'] 
    result['population_churn_share'] = result['population_share'] * result['churn_yes_pct'] 
    return result 


# Adding feature_engineering function 
def feature_engineering(x): 
    """ 
    input: dataframe x 
    - returns a copy containing engineered features that were added 
    (Not dropping existing columns) 
    Should be used before ColumnTransformer so new numeric features are included
    """
    x = x.copy() 
    # Preventing 0 division in case TotalCharges is zero for new customer 
    x['AvgChargesPerMonth'] = x['TotalCharges'] / (x['tenure'] + 1e-6) 
    # Boolean flags 
    x['fiber_internet'] = (x['InternetService'] == 'Fiber optic').astype(int) 
    x['electronic_check'] = (x['PaymentMethod'] == 'Electronic check').astype(int) 
    x['month_to_month'] = (x['Contract'] == 'Month-to-month').astype(int) 
    # Counting the number of internet services being utilized 
    internet_columns = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    count = np.zeros(len(x), dtype=int) 
    for c in internet_columns: 
        if c in x.columns: 
            count += (x[c] == 'Yes').astype(int) 
    x['internet_services_count'] = count 
    return x 


