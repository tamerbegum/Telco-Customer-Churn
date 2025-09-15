# 📊 Customer Churn Prediction with Machine Learning in R

This repository contains an **end-to-end machine learning pipeline** in **R** for predicting customer churn using the **Telco Customer Churn dataset**. The project demonstrates data preprocessing, feature engineering, model training, hyperparameter tuning, class imbalance handling (SMOTE), and advanced visualization techniques.

---

## 📖 Overview

Customer churn is a critical business challenge, especially in subscription-based industries.  
This study applies **supervised machine learning models** to predict whether a customer will churn, leveraging:

- **Comprehensive preprocessing & cleaning**  
- **Feature engineering to capture customer behavior**  
- **Multiple ML algorithms (Logistic Regression, KNN, CART, Random Forest, SVM, MLP, XGBoost)**  
- **Class imbalance solutions (SMOTE oversampling)**  
- **Model interpretability via class maps and feature importance**  

---

## 🔁 Methodology

### 🔹 Data Preprocessing
- Removal of duplicates, irrelevant features (`customerID`)  
- Conversion of categorical features into numeric form (label encoding, one-hot encoding)  
- Handling of missing values and outliers  
- Standardization of numerical variables  

### 🔹 Feature Engineering
- **Contract commitment**: Engaged vs. non-engaged customers  
- **Service bundles**: Total subscribed services, streaming usage flags  
- **Risk signals**: Customers without protection plans, young non-engaged users  
- **Payment habits**: Auto-payment indicator  
- **Tenure segmentation**: Bucketing into yearly intervals  

### 🔹 Model Training
- Data split: **64% training**, **16% validation**, **20% testing**  
- Trained & evaluated models:
  - Logistic Regression  
  - k-Nearest Neighbors (KNN)  
  - Decision Tree (CART)  
  - Random Forest  
  - Multi-Layer Perceptron (MLP)  
  - Support Vector Machine (SVM)  
  - XGBoost (baseline + tuned)  

### 🔹 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC Curve & AUC  

### 🔹 Class Imbalance Handling
- **SMOTE (Synthetic Minority Oversampling Technique)** applied to balance churn (26.5%) vs non-churn (73.5%).  

### 🔹 Visualization
- Correlation heatmaps  
- ROC curves for model comparison  
- Feature importance (XGBoost)  
- Silhouette & class maps  
- Quasi residual plots for key features  

---

## 📂 Dataset

- **Source**: [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Size**: 7,043 customers, 21 features  
- **Target Variable**: `Churn` (Yes = 1, No = 0)  
- **Class Balance**:  
  - 73.5% Non-churn (5,174 customers)  
  - 26.5% Churn (1,869 customers)  

---

## 🚀 Installation

### Required Software
- **R ≥ 4.0**
- **RStudio** recommended for ease of use  

### Required R Packages

```r
packages <- c(
  "dplyr", "ggplot2", "caret", "corrplot", "pROC", 
  "rpart", "randomForest", "StatMatch", "e1071", 
  "xgboost", "nnet", "class", "classmap", "Matrix", 
  "reshape2", "DMwR2", "smotefamily"
)
install.packages(setdiff(packages, rownames(installed.packages())))
```
## 🎯 Usage

Clone this repository:

```bash
git clone https://github.com/your-username/telco-churn-r.git
cd telco-churn-r
```
Open R or RStudio.

Run the script:

```source("churn_analysis.R")```

The script will:
- Clean and preprocess data
- Engineer new features
- Train multiple ML models
- Plot ROC curves & performance comparisons
- Tune XGBoost with cross-validation
- Apply SMOTE balancing and evaluate improvements

## 📁 Project Structure

```bash
├── churn_analysis.R             # Main script (all steps from preprocessing to evaluation)
├── data/
│   └── Telco-Customer-Churn.csv # Dataset (not included, download separately)
├── README.md                    # Project documentation

```

## 📊 Key Results

**Model Performance (Test Set, Original Data)**  
**XGBoost (tuned):**

- AUC ≈ **0.856**  
- Accuracy ≈ **81%**  
- Precision ≈ **0.666**  
- Recall ≈ **0.537**  
- F1 ≈ **0.594**  

**With SMOTE (50/50 balanced training set)**  
**XGBoost + SMOTE:**

- AUC ≈ **0.843**  
- Precision ≈ **0.549**  
- Recall ≈ **0.722**  
- F1 ≈ **0.624**  

➡️ **SMOTE improves recall significantly**, making the model better at detecting churners, though with some precision tradeoff.

---

## 📈 Visualizations

The script generates:

- ROC curves comparing all models  
- Feature importance rankings from XGBoost  
- Silhouette and stacked plots for classification behavior  
- Class maps for churn vs non-churn  
- Quasi residual plots (Tenure, MonthlyCharges, TotalCharges)  

---

## 🔬 Experimental Design

- **Data split**: 64% train, 16% validation, 20% test  
- **Preprocessing**: Label encoding, one-hot encoding, scaling  
- **Hyperparameter tuning**: Grid search for XGBoost  
- **Evaluation protocol**: Multiple metrics (AUC, F1, precision, recall, accuracy)  

---

## 💡 Key Insights

- Longer contract terms and auto-payments reduce churn risk.  
- Customers without protection plans or with high monthly charges are more churn-prone.  
- Tenure is a strong predictor — longer-tenure customers churn less.  
- Initial models suffered from high false negatives (missed churners), which is costly in practice.  
- XGBoost outperforms traditional models in predictive power.  
- SMOTE boosts recall, making the model more effective at identifying churners, even if precision drops slightly.  

  
