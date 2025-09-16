# Install required packages for this project

packages <- c(
  "dplyr",
  "ggplot2",
  "caret",
  "corrplot",
  "pROC",
  "rpart",
  "randomForest",
  "StatMatch",
  "e1071",
  "xgboost",
  "nnet",
  "class",
  "classmap",
  "Matrix",
  "reshape2",
  "DMwR2",
  "smotefamily"
)

# Install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

invisible(lapply(packages, install_if_missing))
