
# preprocessing.py 
# Functions for Pipelines, transformers 

import pandas as pd 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from imblearn.pipeline import Pipeline as ImbPipeline  
from imblearn.over_sampling import SMOTE 
from feature_engineering import feature_engineering 


# Creating a preprocessing pipeline for categorical and numeric columns 
def create_preprocessing_pipeline(data: pd.DataFrame, target_column: str = 'Churn'): 
    # Functions from feature_engineering 
    data = clean_column_names(data) 
    data = handle_missing_values(data) 
    data = feature_engineering(data) 
    # Dropping customerid column 
    data.drop(columns=['customerID'], inplace=True, errors='ignore') 
    # Separate mapping for target column (churn) 
    data[target_column] = data[target_column].map({'Yes':1, 'No':0}) 

    # Column lists 
    numeric_features = ['tenure','MonthlyCharges','TotalCharges','avg_charges_per_month','internet_services_count']
    ohe_features = ['Contract','InternetService','PaymentMethod','tenure_group'] 
    binary_features = ['gender','Partenr','Dependents','PhoneService','PaperlessBilling','MultipleLines','SeniorCitizen',
                       'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    
    binary_mapping = {'Yes':1, 'No':0, 'No phone service':0, 'No internet service':0, 'Male':0, 'Female':1}
    for c in binary_features: 
        if c in data.columns: 
            data[c] = data[c].map(binary_mapping).fillna(data[c]) 

    # Creating transformers 
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('ohe', StandardScaler()) 
    ])   
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ]) 
    # Combining into column transformer 
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features), 
        ('cat', categorical_transformer, ohe_features) 
    ], remainder='drop')  

    return preprocessor, binary_features, binary_mapping 

# Baseline Model Pipelines 
def create_logistic_pipeline(preprocessor, random_state=42): 
    lr_pipeline = Pipeline(steps=[
        ('fe', FunctionTransformer(feature_engineering, validate=False)), 
        ('preproc', preprocessor), 
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)) 
    ])
    return lr_pipeline 

def create_random_forest_pipeline(preprocessor, random_state=42): 
    rf_pipeline = ImbPipeline(steps=[
        ('fe', FunctionTransformer(feature_engineering, validate=False)), 
        ('preproc', preprocessor), 
        ('smote', SMOTE(random_state=random_state)), 
        ('clf', RandomForestClassifier(n_estimators=200, random_state=random_state)) 
    ])
    return rf_pipeline 



