
# src/feature_engineering.py 
# Functions for feature cleaning, feature engineering 

import pandas as pd 
import numpy as np 


# Function for feature engineering 
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame: 
    x = data.copy() 
    # Preventing 0 division in case totalcharges is zero for new customer 
    x['AvgChargesPerMonth'] = x['TotalCharges'] / (x['tenure'] + 1e-6) 
    # Adding boolean flags 
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
    # Creating binary flags for tenure groups 
    x['tenure_group'] = pd.cut(
        x['tenure'], 
        bins=[0, 12, 24, 48, 60, 72], 
        labels=['0-12', '12-24', '24-48', '48-60', '60-72'] 
    )
    return x 




