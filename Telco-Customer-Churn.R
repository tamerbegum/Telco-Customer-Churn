###############################################################################
# SECTION 1: LIBRARIES & GLOBAL SETTINGS
###############################################################################
# Load essential libraries for data processing, visualization, and machine learning

library(dplyr)          # Provides a set of functions to easily manipulate data frames
library(ggplot2)        # Widely used system for creating static data visualizations based on the grammar of graphics
library(caret)          # Provides a unified interface for training and evaluating ML models, cross-validation, preprocessing, etc.
library(corrplot)       # Visualizes correlation matrices in an easy-to-interpret graphical format
library(pROC)           # Tools for visualizing and comparing ROC curves, useful for evaluating classification models
library(rpart)          # Implements recursive partitioning for classification and regression trees (CART)
library(randomForest)   # Implements the Random Forest algorithm for classification and regression
library(StatMatch)      # Used for statistical matching and data integration across sources
library(e1071)          # Includes functions for SVM, Naive Bayes, and other modeling utilities
library(xgboost)        # Extreme Gradient Boosting - efficient and scalable implementation of gradient boosted trees
library(nnet)           # Fits single-hidden-layer neural networks, useful for classification problems
library(class)          # Implements k-Nearest Neighbors (kNN) and other basic classifiers
library(classmap)       # Provides advanced visualizations for classification results (e.g., class maps, silhouette plots)
library(Matrix)         # Provides classes and methods for sparse and dense matrix operations, often needed in ML workflows
library(reshape2)       # Helps in transforming data between wide and long formats (e.g., with `melt()` and `dcast()`)
library(DMwR2)          # Provides tools for regression, classification, and data preprocessing (an updated version of DMwR)

# Global display options to enhance readability in outputs
options(scipen = 999)   # Prevents R from displaying numbers in scientific notation (e.g., 1e+05 → 100000)
options(digits = 3)     # Sets number formatting to show up to 3 decimal places by default

###############################################################################
# SECTION 2: DATA LOADING & INITIAL EXPLORATION
###############################################################################
# Define file path
csv_file <- "/Users/begumtamer/Desktop/Current topics AI/Telco-Customer-Churn.csv"

# Load the dataset
df <- read.csv(csv_file)

# Show first few rows of the dataset
head(df)

# Display the shape of the dataset: 7043 rows, 21 columns (original dataset)
dim(df) # 7043   21

# Check the structure and summary statistics of the dataset to identify data types and potential issues
str(df)
summary(df)

# Remove the 'customerID' column since it is not needed for the analysis
df <- df[, !colnames(df) %in% "customerID"]

# Convert the 'Churn' column to a binary variable (1 for Yes, 0 for No)
df$Churn <- ifelse(df$Churn == "Yes", 1, 0)

# Convert 'TotalCharges' to numeric since it might be read as character initially
df$TotalCharges <- as.numeric(as.character(df$TotalCharges))

# Check the balance of the target variable: count and proportion of churned vs non-churned customers
table(df$Churn)
prop.table(table(df$Churn))
# 0    1 
# 5174 1869 
# 0.735 0.265 

###############################################################################
# SECTION 3: COLUMN-TYPE FUNCTIONS & OUTLIER / MISSING-VALUE HANDLING
###############################################################################
# Function to identify numeric and categorical columns, including those that need special treatment
grab_col_names <- function(dataframe, cat_th = 10, car_th = 20) {
  col_types <- sapply(dataframe, class)
  
  # Identify categorical columns (either character or factor)
  cat_cols <- names(col_types[col_types %in% c("character", "factor")])
  
  # Identify numerical columns that should be treated as categorical (few unique values)
  num_but_cat <- names(dataframe)[
    sapply(dataframe, function(x) length(unique(x)) < cat_th && 
             !is.character(x) && !is.factor(x))
  ]
  
  # Identify categorical columns that are actually cardinal (many unique values)
  cat_but_car <- names(dataframe)[
    sapply(dataframe, function(x) length(unique(x)) > car_th && 
             (is.character(x) || is.factor(x)))
  ]
  
  # Combine and adjust categorical column lists
  cat_cols <- c(cat_cols, num_but_cat)
  cat_cols <- cat_cols[!cat_cols %in% cat_but_car]
  
  # Identify purely numerical columns
  num_cols <- names(col_types[!col_types %in% c("character", "factor")])
  num_cols <- num_cols[!num_cols %in% num_but_cat]
  
  cat_cols <- unique(cat_cols)
  num_cols <- unique(num_cols)
  cat_but_car <- unique(cat_but_car)
  num_but_cat <- unique(num_but_cat)
  
  # Print a summary of the column classification
  cat("Observations:", nrow(dataframe), "\n")
  cat("Variables:", ncol(dataframe), "\n")
  cat("cat_cols:", length(cat_cols), "\n")
  cat("num_cols:", length(num_cols), "\n")
  cat("cat_but_car:", length(cat_but_car), "\n")
  cat("num_but_cat:", length(num_but_cat), "\n")
  
  return(list(cat_cols = cat_cols, num_cols = num_cols, cat_but_car = cat_but_car))
}

# Get the column names based on their types
col_names <- grab_col_names(df)
cat_cols <- col_names$cat_cols
num_cols <- col_names$num_cols
cat_but_car <- col_names$cat_but_car

# Functions for outlier handling: determining thresholds and checking for outliers
outlier_thresholds <- function(dataframe, col_name, q1 = 0.05, q3 = 0.95) {
  quartile1 <- quantile(dataframe[[col_name]], q1, na.rm = TRUE)
  quartile3 <- quantile(dataframe[[col_name]], q3, na.rm = TRUE)
  interquantile_range <- quartile3 - quartile1
  up_limit <- quartile3 + 1.5 * interquantile_range
  low_limit <- quartile1 - 1.5 * interquantile_range
  return(c(low_limit, up_limit))
}

check_outlier <- function(dataframe, col_name) {
  limits <- outlier_thresholds(dataframe, col_name)
  low_limit <- limits[1]
  up_limit <- limits[2]
  any(dataframe[[col_name]] > up_limit | dataframe[[col_name]] < low_limit, na.rm = TRUE)
}

# Check for outliers in all numerical columns
sapply(num_cols, function(col) check_outlier(df, col))
# tenure MonthlyCharges   TotalCharges 
# FALSE          FALSE          FALSE  
# no outliers

# Function to generate a table of missing values with counts and percentages
missing_values_table <- function(dataframe, na_name = FALSE) {
  na_columns <- names(dataframe)[colSums(is.na(dataframe)) > 0]
  n_miss <- colSums(is.na(dataframe[, na_columns, drop = FALSE]))
  ratio <- colSums(is.na(dataframe[, na_columns, drop = FALSE])) / nrow(dataframe) * 100
  
  missing_df <- data.frame(
    n_miss = n_miss,
    ratio = ratio
  )
  
  missing_df <- missing_df[order(-missing_df$n_miss), ]
  
  print(missing_df)
  
  if (na_name) {
    return(na_columns)
  }
}

# Identify columns with missing values and print the missing values table
na_columns <- missing_values_table(df, na_name = TRUE)

# Remove rows with missing values (only 11 rows in the dataset)
df <- na.omit(df)
df <- as.data.frame(df)

# Check again for missing values after removal
missing_values_table(df)
dim(df) # 7032   20

# Check for duplicate rows in the dataset
sum(duplicated(df))  # 22

# Remove duplicate rows
df <- df[!duplicated(df), ] 

# Verify that duplicates have been removed
sum(duplicated(df)) 

# Check dimension again to see how many rows remaining
dim(df) # 7010   20

###############################################################################
# SECTION 4: CATEGORICAL SUMMARIES & CORRELATIONS
###############################################################################
# Function to summarize a categorical column: counts and percentages of each level
cat_summary <- function(dataframe, col_name, plot = FALSE) {
  tab <- table(dataframe[[col_name]])
  prop_tab <- prop.table(tab) * 100
  summary_df <- data.frame(
    Count = as.vector(tab),
    Ratio = as.vector(prop_tab)
  )
  rownames(summary_df) <- names(tab)
  print(summary_df)
  cat("##########################################\n")
  
  if (plot) {
    barplot(tab, main = col_name)
  }
}

# Display summaries for all categorical columns in the dataset
invisible(lapply(cat_cols, function(col) cat_summary(df, col)))

# Calculate and plot the correlation matrix for numerical features
correlation_matrix <- cor(df[, num_cols], use = "complete.obs")
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black", 
         number.cex = 0.7, col = colorRampPalette(c("blue", "white", "red"))(200))

###############################################################################
# SECTION 5: SPLITTING DATA INTO TRAIN, VALIDATION, AND TEST SETS
###############################################################################
# The dataset is split into three sets for training, validation, and testing.
# Training set: 64% of the data.
# Validation set: ~16% of the data.
# Test set: ~20% of the data.
set.seed(0)    # Set seed for reproducibility
set.seed(123)  # Additional seed setting for reproducibility

# Create training indices for 64% of the data
train_indices <- sample(1:nrow(df), size = round(0.64 * nrow(df)), replace = FALSE)

# The remaining indices (36%) are split into validation and test sets
remaining_indices <- setdiff(1:nrow(df), train_indices)
val_size <- round(0.16 * nrow(df))
val_indices <- sample(remaining_indices, size = val_size, replace = FALSE)
test_indices <- setdiff(remaining_indices, val_indices)

###############################################################################
# SECTION 6: FEATURE ENGINEERING
###############################################################################
# This section creates new features from existing columns.
# These engineered features are expected to increase the predictive power of the models.

# Feature 1: NEW_TENURE_YEAR
# -------------------------
# The 'tenure' variable, which is monthly, is segmented into years to capture
# different customer retention periods. The intervals chosen represent 1-year periods.
# The cut() function is used to create factor levels for each interval.
df$NEW_TENURE_YEAR <- cut(df$tenure, 
                          breaks = c(-Inf, 12, 24, 36, 48, 60, 72), 
                          labels = c("0-1 Year", "1-2 Year", "2-3 Year", 
                                     "3-4 Year", "4-5 Year", "5-6 Year"))

# Feature 2: NEW_noProt
# ----------------------
# This binary feature indicates whether a customer lacks protection plans.
df$NEW_noProt <- ifelse(df$OnlineBackup != "Yes" | 
                          df$DeviceProtection != "Yes" | 
                          df$TechSupport != "Yes", 1, 0)
# Customers without multiple protection plans may be at higher risk of churn due to less comprehensive service coverage.

# Feature 3: NEW_Engaged
# -----------------------
# This feature identifies customers who have contract durations either "One year" or "Two year".
# A binary flag is created: 1 for engaged customers and 0 for others.
df$NEW_Engaged <- ifelse(df$Contract %in% c("One year", "Two year"), 1, 0)
# Longer contracts often correlate with increased customer commitment and may lower churn risk.

# Feature 4: NEW_FLAG_AutoPayment
# -------------------------------
# This binary feature marks whether a customer uses an automatic payment method.
# It assigns 1 if the PaymentMethod is either "Bank transfer (automatic)" or "Credit card (automatic)".
df$NEW_FLAG_AutoPayment <- ifelse(df$PaymentMethod %in% c("Bank transfer (automatic)", 
                                                          "Credit card (automatic)"), 1, 0)

# Feature 5: NEW_TotalServices
# ----------------------------
# This feature calculates the total number of additional services a customer subscribes to.
# It sums up binary indicators for several services (e.g., PhoneService, InternetService, etc.),
# counting a service as 1 if the customer has it (indicated by "Yes").
service_cols <- c("PhoneService", "InternetService", "OnlineSecurity", 
                  "OnlineBackup", "DeviceProtection", "TechSupport", 
                  "StreamingTV", "StreamingMovies")
df$NEW_TotalServices <- rowSums(df[, service_cols] == "Yes", na.rm = TRUE)

# Feature 6: NEW_FLAG_ANY_STREAMING
# ----------------------------------
# This binary feature indicates if the customer uses any streaming service (either StreamingTV or StreamingMovies).
# It assigns a 1 if either service is "Yes".
df$NEW_FLAG_ANY_STREAMING <- ifelse(df$StreamingTV == "Yes" | df$StreamingMovies == "Yes", 1, 0)

# Feature 7: NEW_Young_Not_Engaged
# -------------------------------
# This feature flags young customers (non-senior citizens) who are not engaged (do not have long contracts).
# A binary flag is created, where 1 indicates a young customer who is not considered engaged.
df$NEW_Young_Not_Engaged <- ifelse(df$NEW_Engaged == 0 & df$SeniorCitizen == 0, 1, 0)

# Display the updated DataFrame and its dimensions after feature engineering
head(df)
dim(df)

# Recalculate column names after feature engineering to update the lists of categorical and numerical features
col_names <- grab_col_names(df)
cat_cols <- col_names$cat_cols
num_cols <- col_names$num_cols
cat_but_car <- col_names$cat_but_car

###############################################################################
# SECTION 7: LABEL ENCODING, STANDARDIZATION, ONE-HOT ENCODING
###############################################################################
# 7a) Label encoding for binary categorical columns
label_encoder <- function(dataframe, binary_col) {
  dataframe[[binary_col]] <- as.numeric(factor(dataframe[[binary_col]])) - 1
  return(dataframe)
}

binary_cols <- names(df)[sapply(df, function(x) length(unique(x)) == 2 && !is.numeric(x))]
binary_cols

for (col in binary_cols) {
  df <- label_encoder(df, col)
}

# Update the list of categorical columns to exclude the newly encoded binary columns and the target variable "Churn"
cat_cols <- setdiff(cat_cols, c(binary_cols, "Churn"))

# 7b) Standardize numerical columns using statistics from the training set
standardize <- function(dataframe, var, train_ind) {
  var_train <- dataframe[train_ind, var]
  var_train_mean <- mean(var_train, na.rm = TRUE)
  var_train_std <- sd(var_train, na.rm = TRUE)
  dataframe[[var]] <- (dataframe[[var]] - var_train_mean) / var_train_std
  return(dataframe[[var]])
}

for (col in num_cols) {
  df[[col]] <- standardize(df, col, train_indices)
}

# 7c) One-hot encoding for the remaining categorical columns
one_hot_encoder <- function(dataframe, categorical_cols, drop_first = FALSE) {
  result_df <- dataframe
  for (col in categorical_cols) {
    if (!col %in% colnames(result_df)) {
      warning(paste("Column", col, "not found in dataframe. Skipping."))
      next
    }
    
    # Convert column to factor and obtain its levels
    result_df[[col]] <- as.factor(result_df[[col]])
    lvls <- levels(result_df[[col]])
    
    # Skip encoding if the column has only one level
    if (length(lvls) <= 1) {
      warning(paste("Column", col, "has only one level. Skipping."))
      next
    }
    
    # If drop_first is TRUE, omit the first level to avoid multicollinearity
    levels_to_encode <- if (drop_first) lvls[-1] else lvls
    
    # Create dummy variables for each level
    for (lvl in levels_to_encode) {
      new_col_name <- paste0(col, "_", lvl)
      result_df[[new_col_name]] <- ifelse(result_df[[col]] == lvl, 1, 0)
    }
  }
  
  # Remove original categorical columns after encoding
  result_df <- result_df[, !colnames(result_df) %in% categorical_cols]
  return(result_df)
}

df <- one_hot_encoder(df, cat_cols, drop_first = TRUE)
head(df)

###############################################################################
# SECTION 8: PREPARE TRAIN/VAL/TEST SPLITS
###############################################################################
# Separate features (X) and target (y)
X <- df[, names(df) != "Churn"]
y <- df$Churn

# Subset the data for training, validation, and testing based on previously defined indices
X_train <- X[train_indices, ]
y_train <- y[train_indices]

X_val <- X[val_indices, ]
y_val <- y[val_indices]

X_test <- X[test_indices, ]
y_test <- y[test_indices]

###############################################################################
# SECTION 9: MODEL TRAINING & EVALUATION
###############################################################################
# Initialize a list to store ROC curves and AUC scores for each model
model_roc_data <- list()

#########################
# 9.1 Logistic Regression
#########################
print("LR")
# Combine training features and target into one data frame
lr_data <- data.frame(Churn = y_train, X_train)

# Ensure all column names are syntactically valid (e.g., no spaces or special characters)
names(lr_data) <- make.names(names(lr_data))

# Fit a logistic regression model using all predictors
lr_model <- glm(Churn ~ ., data = lr_data, family = "binomial")

X_test_lr <- X_test
names(X_test_lr) <- make.names(names(X_test_lr))

y_probs_lr <- predict(lr_model, X_test_lr, type = "response")

# Convert probabilities to class predictions using a 0.5 threshold
y_pred_lr <- ifelse(y_probs_lr > 0.5, 1, 0)

# Compute ROC curve using predicted probabilities
lr_roc <- roc(y_test, y_probs_lr)

# Store false positive rate (1 - specificity), true positive rate (sensitivity), and AUC
model_roc_data[["LR"]] <- list(
  fpr = 1 - lr_roc$specificities,
  tpr = lr_roc$sensitivities,
  auc = auc(lr_roc)
)

cat("roc_auc score:", model_roc_data[["LR"]]$auc, "\n")
cat("f1 score:", 2 * sum(y_pred_lr & y_test) / (sum(y_pred_lr) + sum(y_test)), "\n")
cat("precision score:", sum(y_pred_lr & y_test) / sum(y_pred_lr), "\n")
cat("recall score:", sum(y_pred_lr & y_test) / sum(y_test), "\n")
cat("accuracy score:", mean(y_pred_lr == y_test), "\n\n")

# roc_auc score: 0.854 
# f1 score: 0.594 
# precision score: 0.662 
# recall score: 0.539 
# accuracy score: 0.813 

#########################
# 9.2 KNN
#########################
print("KNN")
X_train_knn <- as.matrix(X_train)
X_test_knn <- as.matrix(X_test)

knn_model <- knn(train = X_train_knn, test = X_test_knn, cl = y_train, k = 5, prob = TRUE)
y_pred_knn <- as.numeric(as.character(knn_model))

# Extract probabilities from the knn model attribute
prob_attr <- attr(knn_model, "prob")
# Ensure probabilities correspond to class 1
y_probs_knn <- ifelse(y_pred_knn == 1, prob_attr, 1 - prob_attr)

knn_roc <- roc(y_test, y_probs_knn)
model_roc_data[["KNN"]] <- list(
  fpr = 1 - knn_roc$specificities,
  tpr = knn_roc$sensitivities,
  auc = auc(knn_roc)
)

cat("roc_auc score:", model_roc_data[["KNN"]]$auc, "\n")
cat("f1 score:", 2 * sum(y_pred_knn & y_test) / (sum(y_pred_knn) + sum(y_test)), "\n")
cat("precision score:", sum(y_pred_knn & y_test) / sum(y_pred_knn), "\n")
cat("recall score:", sum(y_pred_knn & y_test) / sum(y_test), "\n")
cat("accuracy score:", mean(y_pred_knn == y_test), "\n\n")

# roc_auc score: 0.79 
# f1 score: 0.544 
# precision score: 0.545 
# recall score: 0.542 
# accuracy score: 0.769 

#########################
# 9.3 Decision Tree (CART)
#########################
print("CART")
cart_data <- data.frame(Churn = y_train, X_train)
names(cart_data) <- make.names(names(cart_data))

cart_model <- rpart(Churn ~ ., data = cart_data, method = "class")

X_test_cart <- X_test
names(X_test_cart) <- make.names(names(X_test_cart))
test_data_cart <- data.frame(X_test_cart)

y_probs_cart <- predict(cart_model, test_data_cart)[, 2]
y_pred_cart <- ifelse(y_probs_cart > 0.5, 1, 0)

cart_roc <- roc(y_test, y_probs_cart)
model_roc_data[["CART"]] <- list(
  fpr = 1 - cart_roc$specificities,
  tpr = cart_roc$sensitivities,
  auc = auc(cart_roc)
)

cat("roc_auc score:", model_roc_data[["CART"]]$auc, "\n")
cat("f1 score:", 2 * sum(y_pred_cart & y_test) / (sum(y_pred_cart) + sum(y_test)), "\n")
cat("precision score:", sum(y_pred_cart & y_test) / sum(y_pred_cart), "\n")
cat("recall score:", sum(y_pred_cart & y_test) / sum(y_test), "\n")
cat("accuracy score:", mean(y_pred_cart == y_test), "\n\n")

# roc_auc score: 0.812 
# f1 score: 0.565 
# precision score: 0.703 
# recall score: 0.472 
# accuracy score: 0.815 

#########################
# 9.4 Random Forest
#########################
print("RF")
rf_model <- randomForest(x = as.matrix(X_train), y = as.factor(y_train), ntree = 100)
y_probs_rf <- predict(rf_model, as.matrix(X_test), type = "prob")[, 2]
y_pred_rf <- ifelse(y_probs_rf > 0.5, 1, 0)

rf_roc <- roc(y_test, y_probs_rf)
model_roc_data[["RF"]] <- list(
  fpr = 1 - rf_roc$specificities,
  tpr = rf_roc$sensitivities,
  auc = auc(rf_roc)
)

cat("roc_auc score:", model_roc_data[["RF"]]$auc, "\n")
cat("f1 score:", 2 * sum(y_pred_rf & y_test) / (sum(y_pred_rf) + sum(y_test)), "\n")
cat("precision score:", sum(y_pred_rf & y_test) / sum(y_pred_rf), "\n")
cat("recall score:", sum(y_pred_rf & y_test) / sum(y_test), "\n")
cat("accuracy score:", mean(y_pred_rf == y_test), "\n\n")

# roc_auc score: 0.845 
# f1 score: 0.566
# precision score: 0.649
# recall score: 0.503
# accuracy score: 0.805

#########################
# 9.5 Multi-Layer Perceptron (MLP)
#########################
print("MLP")
X_train_matrix <- as.matrix(X_train)
y_train_matrix <- as.matrix(y_train)
X_test_matrix <- as.matrix(X_test)

# Train the MLP model with a hidden layer of size 10 and decay for convergence
mlp_model <- nnet(X_train_matrix, y_train_matrix, size = 10, maxit = 1000, 
                  decay = 0.1, linout = FALSE, trace = FALSE)

y_probs_mlp <- predict(mlp_model, X_test_matrix)
y_pred_mlp <- ifelse(y_probs_mlp > 0.5, 1, 0)

mlp_roc <- roc(y_test, as.vector(y_probs_mlp))
model_roc_data[["MLP"]] <- list(
  fpr = 1 - mlp_roc$specificities,
  tpr = mlp_roc$sensitivities,
  auc = auc(mlp_roc)
)

cat("roc_auc score:", model_roc_data[["MLP"]]$auc, "\n")
cat("f1 score:", 2 * sum(y_pred_mlp & y_test) / (sum(y_pred_mlp) + sum(y_test)), "\n")
cat("precision score:", sum(y_pred_mlp & y_test) / sum(y_pred_mlp), "\n")
cat("recall score:", sum(y_pred_mlp & y_test) / sum(y_test), "\n")
cat("accuracy score:", mean(y_pred_mlp == y_test), "\n\n")

# roc_auc score: 0.817
# f1 score: 0.569
# precision score: 0.586
# recall score: 0.553 
# accuracy score: 0.787

#########################
# 9.6 SVM
#########################
print("SVM")
svm_model <- svm(x = as.matrix(X_train), y = as.factor(y_train), probability = TRUE)
svm_probs <- predict(svm_model, as.matrix(X_test), probability = TRUE)
y_probs_svm <- attr(svm_probs, "probabilities")[, "1"]
y_pred_svm <- ifelse(y_probs_svm > 0.5, 1, 0)

svm_roc <- roc(y_test, y_probs_svm)
model_roc_data[["SVM"]] <- list(
  fpr = 1 - svm_roc$specificities,
  tpr = svm_roc$sensitivities,
  auc = auc(svm_roc)
)

cat("roc_auc score:", model_roc_data[["SVM"]]$auc, "\n")
cat("f1 score:", 2 * sum(y_pred_svm & y_test) / (sum(y_pred_svm) + sum(y_test)), "\n")
cat("precision score:", sum(y_pred_svm & y_test) / sum(y_pred_svm), "\n")
cat("recall score:", sum(y_pred_svm & y_test) / sum(y_test), "\n")
cat("accuracy score:", mean(y_pred_svm == y_test), "\n\n")

# roc_auc score: 0.813 
# f1 score: 0.574
# precision score: 0.689
# recall score: 0.492 
# accuracy score: 0.815 

#########################
# 9.7 XGBoost
#########################
print("XGBoost")
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(test = dtest),
  early_stopping_rounds = 10,
  verbose = 0,
  print_every_n = 0
)

y_probs_xgb <- predict(xgb_model, dtest)
y_pred_xgb <- ifelse(y_probs_xgb > 0.5, 1, 0)

xgb_roc <- roc(y_test, y_probs_xgb)
model_roc_data[["XGBoost"]] <- list(
  fpr = 1 - xgb_roc$specificities,
  tpr = xgb_roc$sensitivities,
  auc = auc(xgb_roc)
)

cat("roc_auc score:", model_roc_data[["XGBoost"]]$auc, "\n")
cat("f1 score:", 2 * sum(y_pred_xgb & y_test) / (sum(y_pred_xgb) + sum(y_test)), "\n")
cat("precision score:", sum(y_pred_xgb & y_test) / sum(y_pred_xgb), "\n")
cat("recall score:", sum(y_pred_xgb & y_test) / sum(y_test), "\n")
cat("accuracy score:", mean(y_pred_xgb == y_test), "\n\n")

# roc_auc score: 0.856
# f1 score: 0.594
# precision score: 0.666
# recall score: 0.537
# accuracy score: 0.814

#########################
# Plot the ROC curves for each model
#########################

plot(0:1, 0:1, type = "l", lty = 2, xlab = "False Positive Rate", ylab = "True Positive Rate", 
     main = "Model Comparison - ROC Curves")

colors <- rainbow(length(model_roc_data))
for (i in 1:length(model_roc_data)) {
  name <- names(model_roc_data)[i]
  fpr <- model_roc_data[[name]]$fpr
  tpr <- model_roc_data[[name]]$tpr
  auc_score <- model_roc_data[[name]]$auc
  
  lines(fpr, tpr, col = colors[i], lwd = 2)
}

legend("bottomright",
       inset = c(0.02, 0.02),
       legend = paste0(names(model_roc_data), " (AUC = ",
                       round(sapply(model_roc_data, function(x) x$auc), 3), ")"),
       col = colors, lwd = 2,
       cex = 0.7,
       bty = "n")

###############################################################################
# SECTION 10: HYPERPARAMETER TUNING FOR XGBOOST
###############################################################################
# With the highest AUC, XGBoost is selected for further tuning
set.seed(123)  # Set seed for reproducibility

# Prepare DMatrix objects for training and validation sets
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dvalid <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)

# Define a grid of hyperparameters to tune
param_grid <- expand.grid(
  max_depth = c(3, 4, 5),
  eta = c(0.01, 0.1, 0.3),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.8, 1),
  colsample_bytree = c(0.8, 1)
)

# Initialize variables to store the best parameters and best validation AUC score
best_auc <- 0
best_params <- list()
best_nrounds <- 0

# Loop through each combination of hyperparameters
for(i in 1:nrow(param_grid)) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = param_grid$max_depth[i],
    eta = param_grid$eta[i],
    min_child_weight = param_grid$min_child_weight[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i]
  )
  
  # Train model on the training set and evaluate on the validation set
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 200,         # Maximum rounds
    watchlist = list(val = dvalid),
    early_stopping_rounds = 10,
    verbose = 0,
    maximize = TRUE
  )
  
  # Record the best validation AUC from this model
  current_auc <- model$best_score
  
  # Update best parameters if current model outperforms previous models
  if(current_auc > best_auc) {
    best_auc <- current_auc
    best_params <- params
    best_nrounds <- model$best_iteration
  }
  
  cat("Validation AUC:", current_auc, "with params:", 
      paste(names(params), params, sep = "=", collapse = ", "), "\n")
}

cat("Best Validation AUC:", best_auc, "\n")
cat("Best Parameters:", paste(names(best_params), best_params, sep = "=", collapse = ", "), "\n")
cat("Best number of rounds:", best_nrounds, "\n")
# Best Parameters: objective=binary:logistic, eval_metric=auc, max_depth=4, eta=0.3, min_child_weight=5, subsample=1, colsample_bytree=1

# Retrain the best model using the best parameters on the training set
best_model <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

# Predict on the test set with the tuned model
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)
y_probs_xgb_tuned <- predict(best_model, dtest)
y_pred_xgb_tuned <- ifelse(y_probs_xgb_tuned > 0.5, 1, 0)

# Evaluate performance of the tuned model on the test set
xgb_tuned_roc <- roc(y_test, y_probs_xgb_tuned)
tuned_auc <- auc(xgb_tuned_roc)
cat("Tuned XGBoost AUC on Test Set:", tuned_auc, "\n")

# Plot the ROC curve for the tuned XGBoost model
plot(xgb_tuned_roc,
     identity = FALSE,
     col = "blue",
     main = "ROC Curve for XGBoost Model",
     print.auc = TRUE)

###############################################################################
# SECTION 11: FEATURE IMPORTANCE
###############################################################################
# Calculate the feature importance matrix for the best tuned XGBoost model
importance_matrix <- xgb.importance(model = best_model)
print(importance_matrix)

# Plot the feature importance (scaled relative to the most important feature)
xgb.plot.importance(importance_matrix = importance_matrix,
                    main = "XGBoost Feature Importance",
                    rel_to_first = TRUE,
                    xlab = "Relative Importance")

###############################################################################
# SECTION 12: CLASSMAP-BASED VISUALIZATIONS (PAC, SILHOUETTE, STACKED, ETC.)
###############################################################################
# Compute PAC (Probability of Alternative Class) for each test case:
# For Churn = 1, PAC is 1 - predicted probability; for Churn = 0, it is the predicted probability.
PAC_xgb <- ifelse(y_test == 1, 1 - y_probs_xgb, y_probs_xgb)

# Compute silhouette values s(i):
# For PAC <= 0.5, s(i) = 1 - 2*PAC; otherwise, s(i) = 2*PAC - 1.
s_xgb <- ifelse(PAC_xgb <= 0.5, 1 - 2 * PAC_xgb, 2 * PAC_xgb - 1)

# Create a dummy object 'vcrout_xgb' for silhouette plotting
vcrout_xgb <- list(
  PAC = PAC_xgb,
  s = s_xgb,
  yint = ifelse(y_test == 1, 2, 1),
  levels = c("No Churn", "Churn"),
  classLabels = c("No Churn", "Churn")
)

# Silhouette Plot
silplot(vcrout_xgb, 
        main = "Silhouette Plot", 
        summary = TRUE)

# For the stacked plot, create a minimal 'vcrout_xgb' with true and predicted labels.
true_labels <- factor(ifelse(y_test == 1, "Churn", "No Churn"),
                      levels = c("No Churn", "Churn"))
pred_labels <- factor(ifelse(y_pred_xgb == 1, "Churn", "No Churn"),
                      levels = c("No Churn", "Churn"))

vcrout_xgb <- list(
  y    = true_labels,
  yint = as.numeric(true_labels),
  pred = pred_labels,
  predint = as.numeric(pred_labels),
  levels = c("No Churn", "Churn"),
  ofarness = rep(0, length(true_labels)) # Dummy values
)

# Stacked Plot for visualizing the confusion matrix
stackedplot(
  vcrout_xgb,
  main       = "Stacked Plot",
  showLegend = TRUE,
  htitle     = "Given Class",
  vtitle     = "Predicted Class"
)

# Quasi Residual Plots for visualizing model residuals against features
# Example with the 'tenure' feature
feat <- df$tenure[test_indices]
qresplot(
  PAC_xgb,
  feat,
  xlab         = "Tenure",
  main         = "Quasi Residual Plot - Tenure",
  gray         = TRUE,
  plotQuantiles = TRUE,       
  probs        = c(0.25, 0.5, 0.75), 
  cols         = c("blue", "green", "red"),  
  opacity      = 0.6
)

# Example with the 'MonthlyCharges' feature
feat <- df$MonthlyCharges[test_indices]
qresplot(
  PAC_xgb,
  feat,
  xlab         = "MonthlyCharges",
  main         = "Quasi Residual Plot - MonthlyCharges",
  gray         = TRUE,
  plotQuantiles = TRUE,       
  probs        = c(0.25, 0.5, 0.75), 
  cols         = c("blue", "green", "red"),  
  opacity      = 0.6
)

# Example with the 'TotalCharges' feature
feat <- df$MonthlyCharges[test_indices]
qresplot(
  PAC_xgb,
  feat,
  xlab         = "TotalCharges",
  main         = "Quasi Residual Plot - TotalCharges",
  gray         = TRUE,
  plotQuantiles = TRUE,       
  probs        = c(0.25, 0.5, 0.75), 
  cols         = c("blue", "green", "red"),  
  opacity      = 0.6
)

# Class Map for Churn and No Churn
givenlab <- factor(
  ifelse(y_test == 1, "Churn", "No Churn"),
  levels = c("No Churn", "Churn")
)
predlab <- factor(
  ifelse(y_probs_xgb > 0.5, "Churn", "No Churn"),
  levels = c("No Churn", "Churn")
)
givenint <- as.numeric(givenlab)
predint  <- as.numeric(predlab)

# Probability of the alternative class for class maps
PAC_xgb <- ifelse(y_test == 1, 1 - y_probs_xgb, y_probs_xgb)
altlab <- ifelse(givenlab == "Churn", "No Churn", "Churn")
altint <- ifelse(givenlab == "Churn", 1, 2)

# Prepare feature matrix and scale it
X_test_scaled <- scale(X_test)  # Scale the test features

# Compute Mahalanobis distances for each class
# Add regularization to prevent singular covariance matrices
cov_churn <- cov(X_test_scaled[y_test == 1, ])
cov_nochurn <- cov(X_test_scaled[y_test == 0, ])
diag(cov_churn) <- diag(cov_churn) + 0.001
diag(cov_nochurn) <- diag(cov_nochurn) + 0.001

farness_churn <- mahalanobis(X_test_scaled,
                             colMeans(X_test_scaled[y_test == 1, ]),
                             cov_churn)

farness_nochurn <- mahalanobis(X_test_scaled,
                               colMeans(X_test_scaled[y_test == 0, ]),
                               cov_nochurn)

# Combine into proper farness measure
farnessVal <- ifelse(y_test == 1, farness_churn, farness_nochurn)

# Scale farness to [0,1] range for better visualization
farnessVal <- (farnessVal - min(farnessVal)) / (max(farnessVal) - min(farnessVal))

# Create proper figMatrix (using PCA scores)
pca_scores <- prcomp(X_test_scaled)$x[, 1:2]  # First 2 principal components
figMatrix <- pca_scores

# Build vcrout object
vcrout_xgb <- list(
  y = givenlab,
  yint = givenint,
  pred = predlab,
  predint = predint,
  altlab = altlab,
  altint = altint,
  PAC = PAC_xgb,
  fig = figMatrix,  # Now contains spatial coordinates
  farness = farnessVal,
  ofarness = farnessVal,
  levels = c("No Churn", "Churn")
)

# Generate corrected class maps
classmap(
  vcrout_xgb,
  whichclass = "Churn",
  main = "Class Map for Churn",
  cutoff = 0.95,
  plotcutoff = TRUE,
  opacity = 0.8
)

classmap(
  vcrout_xgb,
  whichclass = "No Churn",
  main = "Class Map for No Churn",
  cutoff = 0.95,
  plotcutoff = TRUE,
  opacity = 0.8
)

# Display summary statistics
summary(y_probs_xgb)

# Create a histogram to visualize the distribution
hist(y_probs_xgb, 
     breaks = 30, 
     main = "Distribution of y_probs_xgb", 
     xlab = "Predicted Probability", 
     col = "lightblue")

###############################################################################
# SECTION 13: SMOTE IMPLEMENTATION FOR XGBOOST WITH 50/50 BALANCE
###############################################################################

# Ensure necessary libraries are loaded
library(smotefamily)

# Set seed for reproducibility
set.seed(123)
# Prepare training data frame
X_train_df <- as.data.frame(X_train)
X_train_df$Churn <- factor(y_train, levels = c(0, 1))

# Compute original class counts
n0 <- sum(y_train == 0)
n1 <- sum(y_train == 1)
# Determine duplication size so that minority >= majority
dup_size <- ceiling(n0 / n1 - 1)

# Apply SMOTE to oversample the minority class
smote_out <- SMOTE(
  X = X_train_df[, -ncol(X_train_df)],
  target = X_train_df$Churn,
  K = 5,
  dup_size = dup_size
)
smote_data <- smote_out$data
names(smote_data)[ncol(smote_data)] <- "Churn"

# Down‑sample each class to exactly n0 rows for 50/50 balance
balanced_smote <- smote_data %>%
  group_by(Churn) %>%
  sample_n(n0) %>%
  ungroup()

cat("Balanced class distribution:\n")
print(table(balanced_smote$Churn))  
# 0    1 
# 3293 3293 

# Prepare SMOTE‑balanced feature matrix and target vector
X_train_smote <- balanced_smote[, setdiff(names(balanced_smote), "Churn")]
y_train_smote <- as.numeric(as.character(balanced_smote$Churn))

# Build DMatrix objects
dtrain_smote <- xgb.DMatrix(data = as.matrix(X_train_smote), label = y_train_smote)
dtest_smote  <- xgb.DMatrix(data = as.matrix(X_test),        label = y_test)

# Train XGBoost with SMOTE-applied data using best params/rounds
xgb_smote_model <- xgb.train(
  params  = best_params,
  data    = dtrain_smote,
  nrounds = best_nrounds,
  verbose = 0
)

# Predict on test set
y_probs_xgb_smote <- predict(xgb_smote_model, dtest_smote)
y_pred_xgb_smote  <- ifelse(y_probs_xgb_smote > 0.5, 1, 0)

# Compute ROC & AUC
xgb_smote_roc <- pROC::roc(y_test, y_probs_xgb_smote)
smote_auc     <- pROC::auc(xgb_smote_roc) 
cat("\nSMOTE XGBoost AUC on Test Set:", round(smote_auc, 3), "\n") # 0.843
cat("Original XGBoost AUC:",      round(tuned_auc, 3),   "\n\n") 

# Classification report
conf_matrix <- table(Predicted = y_pred_xgb_smote, Actual = y_test)
cat("Confusion Matrix:\n")
print(conf_matrix)

precision <- conf_matrix[2,2] / sum(conf_matrix[2,])
recall    <- conf_matrix[2,2] / sum(conf_matrix[,2])
f1_score  <- 2 * (precision * recall) / (precision + recall)

cat("\nPrecision:", round(precision, 3), "\n") # 0.549 
cat("Recall:",    round(recall,    3), "\n") # 0.722 
cat("F1 Score:",  round(f1_score,  3), "\n") # 0.624 

###############################################################################
# SECTION 14: CLASSMAP VISUALIZATIONS FOR SMOTE XGBOOST
###############################################################################

# Compute PAC (Probability of Alternative Class) for SMOTE model
PAC_xgb_smote <- ifelse(y_test == 1, 1 - y_probs_xgb_smote, y_probs_xgb_smote)

# Compute silhouette values
s_xgb_smote <- ifelse(PAC_xgb_smote <= 0.5, 1 - 2 * PAC_xgb_smote, 2 * PAC_xgb_smote - 1)

# Create vcrout object for SMOTE model
true_labels <- factor(ifelse(y_test == 1, "Churn", "No Churn"),
                      levels = c("No Churn", "Churn"))
pred_labels <- factor(ifelse(y_pred_xgb_smote == 1, "Churn", "No Churn"),
                      levels = c("No Churn", "Churn"))

vcrout_xgb_smote <- list(
  y = true_labels,
  yint = as.numeric(true_labels),
  pred = pred_labels,
  predint = as.numeric(pred_labels),
  PAC = PAC_xgb_smote,
  s = s_xgb_smote,
  levels = c("No Churn", "Churn"),
  ofarness = rep(0, length(true_labels)) # Dummy values
)

# Silhouette Plot for SMOTE model
silplot(vcrout_xgb_smote, 
        main = "Silhouette Plot (SMOTE XGBoost)", 
        summary = TRUE)

# Stacked Mosaic Plot for SMOTE model
stackedplot(
  vcrout_xgb_smote,
  main       = "Stacked Mosaic Plot (SMOTE XGBoost)",
  showLegend = TRUE,
  htitle     = "Given Class",
  vtitle     = "Predicted Class"
)

# Class Maps for SMOTE model
# Prepare feature matrix and scale it
X_test_scaled <- scale(X_test)

# Compute Mahalanobis distances with regularization
cov_churn <- cov(X_test_scaled[y_test == 1, ])
cov_nochurn <- cov(X_test_scaled[y_test == 0, ])
diag(cov_churn) <- diag(cov_churn) + 0.001
diag(cov_nochurn) <- diag(cov_nochurn) + 0.001

farness_churn <- mahalanobis(X_test_scaled,
                             colMeans(X_test_scaled[y_test == 1, ]),
                             cov_churn)

farness_nochurn <- mahalanobis(X_test_scaled,
                               colMeans(X_test_scaled[y_test == 0, ]),
                               cov_nochurn)

farnessVal <- ifelse(y_test == 1, farness_churn, farness_nochurn)
farnessVal <- (farnessVal - min(farnessVal)) / (max(farnessVal) - min(farnessVal))

# PCA scores for visualization
pca_scores <- prcomp(X_test_scaled)$x[, 1:2]

# Update vcrout object with spatial information
vcrout_xgb_smote$fig <- pca_scores
vcrout_xgb_smote$farness <- farnessVal
vcrout_xgb_smote$ofarness <- farnessVal

# Generate class maps
classmap(
  vcrout_xgb_smote,
  whichclass = "Churn",
  main = "Class Map for Churn (SMOTE XGBoost)",
  cutoff = 0.95,
  plotcutoff = TRUE,
  opacity = 0.8
)

classmap(
  vcrout_xgb_smote,
  whichclass = "No Churn",
  main = "Class Map for No Churn (SMOTE XGBoost)",
  cutoff = 0.95,
  plotcutoff = TRUE,
  opacity = 0.8
)

# Compare ROC curves
plot(xgb_smote_roc, col = "blue", main = "ROC Curve Comparison")
plot(xgb_tuned_roc, col = "red", add = TRUE)
legend("bottomright", 
       legend = c(paste("SMOTE XGBoost (AUC =", round(smote_auc, 3), ")"), 
                  paste("Original XGBoost (AUC =", round(tuned_auc, 3), ")")),
       col = c("blue", "red"), lwd = 2)

# Feature importance for SMOTE model
importance_matrix_smote <- xgb.importance(model = xgb_smote_model)
xgb.plot.importance(importance_matrix_smote,
                    main = "XGBoost Feature Importance (SMOTE)",
                    rel_to_first = TRUE,
                    xlab = "Relative Importance")


# Quasi Residual Plot for 'Tenure'
feat <- df$tenure[test_indices]
qresplot(
  PAC_xgb_smote,
  feat,
  xlab         = "Tenure",
  main         = "Quasi Residual Plot - Tenure (SMOTE)",
  gray         = TRUE,
  plotQuantiles = TRUE,
  probs        = c(0.25, 0.5, 0.75),
  cols         = c("blue", "green", "red"),
  opacity      = 0.6
)


# Quasi Residual Plot for 'Monthly Charges'
feat <- df$MonthlyCharges[test_indices]
qresplot(
  PAC_xgb_smote,
  feat,
  xlab         = "Monthly Charges",
  main         = "Quasi Residual Plot - Monthly Charges",
  gray         = TRUE,
  plotQuantiles = TRUE,       
  probs        = c(0.25, 0.5, 0.75), 
  cols         = c("blue", "green", "red"),  
  opacity      = 0.6
)

# Quasi Residual Plot for 'Total Charges'
feat <- df$TotalCharges[test_indices]
qresplot(
  PAC_xgb_smote,
  feat,
  xlab         = "Total Charges",
  main         = "Quasi Residual Plot - Total Charges",
  gray         = TRUE,
  plotQuantiles = TRUE,       
  probs        = c(0.25, 0.5, 0.75), 
  cols         = c("blue", "green", "red"),  
  opacity      = 0.6
)

