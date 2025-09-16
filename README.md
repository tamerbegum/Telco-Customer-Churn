# 📊 Customer Churn Prediction with Machine Learning in R

This repository provides an **end-to-end machine learning pipeline** in **R** for predicting customer churn using the **Telco Customer Churn dataset**.  
It covers data preprocessing, feature engineering, model training, hyperparameter tuning, class imbalance handling (SMOTE), and advanced visualization.

---

## 📖 Overview

Customer churn is a major challenge in subscription-based businesses.  
This study applies **supervised ML models** to predict churn with:

- Thorough **preprocessing & cleaning**  
- **Feature engineering** to capture behavior & risk  
- Multiple algorithms: Logistic Regression, KNN, CART, Random Forest, MLP, SVM, XGBoost  
- **Class imbalance solutions** (SMOTE)  
- **Interpretability tools**: feature importance & class maps  

---

## 🔁 Methodology

### 🔹 Data Preprocessing
- Remove duplicates, irrelevant features (`customerID`)  
- Encode categorical features (label/one-hot)  
- Handle missing values & outliers  
- Standardize numerical variables  

### 🔹 Feature Engineering
- Contract commitment (engaged vs. non-engaged)  
- Service bundles (streaming, total services)  
- Risk signals (no protection plans, young + non-engaged)  
- Payment habits (auto-payment indicator)  
- Tenure segmentation (yearly buckets)  

### 🔹 Model Training & Evaluation
- Data split: **64% train**, **16% validation**, **20% test**  
- Algorithms: Logistic Regression, KNN, CART, Random Forest, MLP, SVM, XGBoost  
- Metrics: Accuracy, Precision, Recall, F1-score, ROC Curve & AUC  

### 🔹 Class Imbalance
- **SMOTE oversampling** to balance churn (26.5%) vs. non-churn (73.5%)  

### 🔹 Visualization
- Correlation heatmaps  
- ROC curves for comparison  
- Feature importance (XGBoost)  
- Advanced plots (class maps, quasi residuals, silhouette, mosaic)  

---

## 📂 Dataset

- **Source**: [Kaggle – Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Size**: 7,043 customers, 21 features  
- **Target**: `Churn` (Yes = 1, No = 0)  
- **Distribution**: 73.5% Non-churn, 26.5% Churn  

---

## 🚀 Installation

### Software
- **R ≥ 4.0**
- **RStudio** recommended  

### R Packages

```r
packages <- c(
  "dplyr", "ggplot2", "caret", "corrplot", "pROC", 
  "rpart", "randomForest", "StatMatch", "e1071", 
  "xgboost", "nnet", "class", "classmap", "Matrix", 
  "reshape2", "DMwR2", "smotefamily"
)
install.packages(setdiff(packages, rownames(installed.packages())))
 ```

## 📦 R Dependencies
Run the following in R to install all required packages:
```r
source("requirements.R")

## 🎯 Usage

Clone this repository:

```bash
git clone https://github.com/your-username/telco-churn-r.git
cd telco-churn-r
```

Open R or RStudio and run:

```r
source("churn_analysis.R")
```

This will:
- Preprocess and clean data  
- Engineer features  
- Train models & tune XGBoost  
- Plot ROC curves and feature importance  
- Apply SMOTE and compare results  

---

## 📁 Project Structure

```bash
├── churn_analysis.R             # Main script
├── data/
│   └── Telco-Customer-Churn.csv # Dataset (download separately)
├── README.md                    # Documentation
```
## 📊 Results

### XGBoost (Tuned, Test Set)
- **AUC**: 0.856  
- **Accuracy**: 81%  
- **Precision**: 0.666  
- **Recall**: 0.537  
- **F1**: 0.594  

### XGBoost + SMOTE (Balanced Training)
- **AUC**: 0.843  
- **Precision**: 0.549  
- **Recall**: 0.722  
- **F1**: 0.624  

➡️ SMOTE significantly improves recall (better detection of churners) at the cost of some precision.  

---

## 💡 Why XGBoost?
- Outperforms Logistic Regression, KNN, CART, Random Forest, MLP, and SVM  
- Handles nonlinear interactions effectively  
- Built-in **feature importance**  
- Robust to overfitting with proper tuning  
- With SMOTE, offers the **best recall** for churn detection  

---

## 📈 Visualizations

The script produces:
- ROC curves across models  
- Feature importance (XGBoost)  
- Advanced diagnostics:  
  - **PAC (Probability of Alternative Class):** Measures how close each prediction is to the opposite class, highlighting uncertain cases.  
  - **Silhouette plots (classification confidence):** Indicate how confidently the model classifies each instance; values near 1 mean high certainty, values near 0 suggest uncertainty, and negative values point to likely misclassifications.  
  - **Mosaic confusion plots:** A visual representation of the confusion matrix that makes it easier to see strengths and weaknesses, especially for minority classes.  
  - **Quasi residual plots (`Tenure`, `MonthlyCharges`, `TotalCharges`):** Show residuals against key features to reveal systematic errors or patterns where the model underperforms.  
  - **Class maps (PCA + Mahalanobis distance for misclassifications):** Plot instances in 2D space to visualize relationships between predicted and actual classes, making misclassifications and outliers more interpretable.  

---

## 🔬 Experimental Design
- **Split**: 64% train / 16% validation / 20% test  
- **Preprocessing**: encoding, scaling  
- **Hyperparameter tuning**: Grid search (XGBoost)  
- **Evaluation**: AUC, F1, precision, recall, accuracy  



