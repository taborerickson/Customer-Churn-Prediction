
# Telco Customer Churn Prediction 
End-to-end project for analyzing customer churn data and predicting customers likely to leave in order to produce targeted, optimized, and actionable customer retention strategies. 

Data source: Telco Customer Churn Dataset from Kaggle 
https://www.kaggle.com/blastchar/telco-customer-churn 


**Goals:** <br> 
1. Analyze Telco customer churn dataset and produce actionable customer retention insights. 
2. Predict customers that are likely to leave the company in order to gain insights into targeted and optimized retention strategies. 
<br></br> 

**Approach:** <br> 
1. Exploratory Data Analysis (EDA) to uncover drivers of customer churn. 
2. Feature engineering (AvgChargesPerMonth, num_internet_services, flags for fiber optic/electronic check/month-to-month). 
3. Pipelines with preprocessing and modeling (Logistic Regression baseline, Random Forest with SMOTE, RandomizedSearchCV tuning).  
4. Model explainability with SHAP for top features. 
<br></br>

**Results:** <br> 
- Best Model: Tuned RandomForest 
    - Selected via RandomizedSearchCV: 
        - CV Best ~0.842 (roc_auc) 
    - Final Test: ROC AUC -> 0.839, PR AUC -> 0.639 
    - Probability Threshold ~0.6036 (approximated by max f1) 
- Classfication at threshold 0.6036 (test_set -> n=1,409): 
    - True negatives (class 0 support = 1,035) 
    - True positives (class 1 support = 374) 
    - Positve-class recall (sensitivity) ~0.72 -> ~269 of 374 churns correctly flagged 
    - False positives ~211 (customers flagged but would not have churned) 
    - False negatives ~105 (actual churn missed by the model) 
- Baselines: 
    - LogisticRegression: ROC AUC -> 0.835, PR AUC -> 0.629 
    - RandomForest: ROC AUC -> 0.803, PR AUC -> 0.576

<br></br> 


**Top Features:**<br>
1. 
<br></br> 

**Top Actionable Recommendations:** 
- (Month-to-Month Customers): Consider short-term discounts to allow contract conversion with upfront incentives (Example: first 3 months 15% off for one-year plans). 
- (Fiber Optic Internet): Investigate service quality such as speed, outages, and customer expectations. Consider a targeted bundled offer including internet service and free/discounted tech support. 
- (Electronic Check Customers): Investigate negative customer billing reviews to identify drivers for the in-group electronic check customer churn. Explore billing process improvements or incentives for alternate payment methods. 
- (New High Monthly Charges Customers [low tenure]): Implement a new customer onboarding program, or new customer assistance/short-term discounts to promote customer satisfaction in new customers 
- (Customers Without Internet Support Services): Explore up-selling opportunity and retention strategies through offering bundled internet support packages (Tech Support/Online Security/Online Backup/Device Protection) (Streaming TV/Streaming Movies).  

**Recommended Actions (low -> high cost)**
- Low-Cost / Automated: 
    - Email + SMS: Send targeted marketing messages highlighting limited-time discounts or service benefits. 
    - In-Product Marketing: Add advertisements for short-term discounts or free-trial-periods for additional service add-ons. 
- Medium-Cost: 
    - Personalized Offers: Have representatives make out-bound calls to high-churn risk customers to offer personalized offers. 
    - Conversion Incentives: Offer contract upgrade incentives to customers on month-to-month contracts to incentivize conversions to longer contracts. 
    - Rewards Program: Implement and advertise a rewards program to reward: customer tenure, low-churn risk payment method selection, and customers with multiple services. 
- High-Cost: 
    - Premium discounts or hardware replacements for strategic customers. 


Files in Repo: 
- Data/     (Contains both raw data file and cleaned data) 
- Models/   (Contains model pipelines) 
- Notebooks/    (Contains notebooks with documentation) 
- Visuals/      (Contains .png figures of EDA visualizations and model performance metrics) 
- Demo_Prediction.ipynb     (Notebook to run a demo prediction on customer churn)  
- README.md         
- requirements.txt  (.txt file listing the project dependencies to install) 

**Note**<br> 
Notebooks *customer_churn_EDA.ipynb* and *customer_churn_ml.ipynb* are still in progress. These notebooks won't provide any additional information, but rather are intended to separate out the Exploratory Data Analysis (EDA) and model pipeline development. 




Limitations and Next Steps: 

Multicollinearity (internet-service + internet-features) 
Correlation vs Causation 
Permutation Importance 
Logistic Coefficient Stability checks 

Implement experimentation models to target customers where interventions truly reduce customer churn

Interaction Features 
Chi-Square 

Implement sentiment analysis model for analyzing negative customer billing reviews to investigate billing experience for customers paying with electronic check. 

A/B testing on different Bundled Internet Support packages for customers. 



**Resume Bullet**<br> 
Built end-to-end customer churn prediction pipeline using Telco dataset — performed feature engineering, handled class imbalance (SMOTE & class weights), tuned XGBoost model (ROC AUC 0.86), and produced SHAP explainability with actionable retention recommendations.


