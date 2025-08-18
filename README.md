
<center> 

# Telco Customer Churn (Prediction & Analysis)

<br></br> 

**Data source: Telco Customer Churn Dataset from Kaggle** 
https://www.kaggle.com/blastchar/telco-customer-churn 

</center> 

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
        - CV Best ~0.8421 (roc_auc) 
    - Final Test: ROC AUC -> 0.8404, PR AUC -> 0.6401 
    - Probability Threshold ~0.5784 (approximated by max f1) 
- Classfication at threshold 0.5784 (test_set -> n=1,409) 
- Baselines: 
    - LogisticRegression: ROC AUC -> 0.0.8368, PR AUC -> 0.0.6369 
    - RandomForest: ROC AUC -> 0.0.8061, PR AUC -> 0.0.5784

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


**Resume Bullet**<br>
Built end-to-end customer churn prediction pipeline using Telco dataset -- performed feature engineering, handled class imbalance (SMOTE & class weights), tuned XGBoost model (ROC AUC 0.86), and produced SHAP explainability with actionable retention recommendations.

Designed and implemented a reproducible end-to-end machine learning pipeline including feature engineering, preprocessing pipelines, class-imbalance handling, baseline and tuned models, and model explainability using SHAP

Telco Customer Churn Predictions 
- Engineered an end-to-end prediction pipeline (data ingestion, data cleaning, feature engineering, preprocessing pipelines, class-imbalance handling, model selection/tuning) using Python, pandas, scikit-learn and imbalanced-learn. 
- Packaged tuned Random Forest model (ROC AUC 0.84 and PR AUC 0.64) with reproducible pipeline and model explainability. 
- Produced actionable insights and operationalized pipeline enabling targeted retention recommendations for high-risk churn candidates. 


- Engineered a robust end-to-end machine learning prediction pipeline, producing a packaged reproducible and operationalized pipeline and model explainability. 
- 
- Developed an end-to-end machine learning prediction pipeline achieving ROC AUC 0.84, producing an interpretable SHAP insights and reproducible and operationalized pipeline. 
- Implemented feature engineering methods, model tuning, class imbalance handling

- Developed and implemented an end-to-end machine learning pipeline using modular code methods including feature engineering, preprocessing pipelines, and class-imbalance handling achieving ROC AUC 0.84 and PR AUC 0.64 on a held-out test set.
- Packaged reproducible and operationalized pipeline and model explainabilty 
a reproducible packaged end-to-end machine learning pipeline including feature engineering, preprocessing pipelines, and class-imbalance handling
- Packaged 
