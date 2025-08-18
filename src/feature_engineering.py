
# src/feature_engineering.py 
# Functions for feature cleaning, feature engineering 

import pandas as pd 
import numpy as np 

# Standardizes column names by stripping spaces and making lowercase 
def clean_column_names(data: pd.DataFrame) -> pd.DataFrame: 
    data.columns = data.columns.str.strip().str.lower().str.replace(' ','_') 
    return data 


# Function for feature engineering 
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame: 
    x = data.copy() 
    # Preventing 0 division in case totalcharges is zero for new customer 
    x['avg_charges_per_month'] = x['totalcharges'] / (x['tenure'] + 1e-6) 
    # Adding boolean flags 
    x['fiber_internet'] = (x['internetservice'] == 'Fiber optic').astype(int) 
    x['electronic_check'] = (x['paymentmethod'] == 'Electronic check').astype(int) 
    x['month_to_month'] = (x['contract'] == 'Month-to-month').astype(int) 
    # Counting the number of internet services being utilized 
    internet_columns = ['onlinesecurity','onlinebackup','deviceprotection','techsupport','streamingtv','streamingmovies']
    count = np.zeros(len(x), dtype=int) 
    for c in internet_columns: 
        if c in x.columns: 
            count += (x[c] == 'Yes').astype(int) 
    x['internet_services_count'] = count 
    # Creating binary flags for tenure groups 
    x['tenure_group'] = pd.cut(
        x['tenure'], 
        bins=[0, 12, 24, 48, 60, 72], 
        labels=['0-12', '12-24', '24-48', '48-60', '60-72'] 
    )
    return x 




