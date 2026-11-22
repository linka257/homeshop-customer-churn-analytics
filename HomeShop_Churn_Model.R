# ==============================================================================
# Project: Customer Churn Prediction at HomeShop
# Author:  Linka Ndizeye (https://github.com)
# Tools:   R (caret, pROC, car, tidyverse) | Paired with Power BI Diagnostics
# Dataset: 24,017 retail customer records across Swiss regions
# ==============================================================================

library(caret)
library(pROC)
library(car)
library(tidyverse)

# 1. Load & Prepare Data -------------------------------------------------------
Homeshop <- read_csv("Homeshop.csv")

# Convert target variable to factor ("No" = Retained, "Yes" = Churned)
Homeshop$Churn <- factor(Homeshop$Churn, levels = c(0, 1), labels = c("No", "Yes"))

# Check class distribution (~70.1% Retained, ~29.9% Churned)
prop.table(table(Homeshop$Churn))

# 2. Train / Test Split (80% Train, 20% Test) ----------------------------------
set.seed(123)
trainIndex <- createDataPartition(Homeshop$Churn, p = 0.8, list = FALSE)
trainData  <- Homeshop[trainIndex, ]
testData   <- Homeshop[-trainIndex, ]

# 3. Logistic Regression Modeling & Feature Selection --------------------------
# Baseline full model (excluding singular DiscountRate)
model_full <- glm(Churn ~ . - DiscountRate, data = trainData, family = binomial)
summary(model_full)

# Check multicollinearity across predictors
vif(model_full)

# Refined model keeping statistically significant behavioral & loyalty drivers
model_refined <- glm(
  Churn ~ NumPurchases + SupportInteractions + MembershipLevel + Month + Region,
  data = trainData,
  family = binomial
)
summary(model_refined)

# Calculate Odds Ratios (OR) for business interpretation
odds_ratios <- exp(coef(model_refined))
print(round(odds_ratios, 3))

# 4. Out-of-Sample Model Evaluation --------------------------------------------
testData$prob_Churn <- predict(model_refined, newdata = testData, type = "response")
testData$pred_class <- factor(ifelse(testData$prob_Churn >= 0.5, "Yes", "No"), levels = c("Yes", "No"))
testData$Churn      <- factor(testData$Churn, levels = c("Yes", "No"))

# Confusion Matrix (Accuracy: 70.7%, Specificity: 94.7%, Sensitivity: 14.6%)
conf_matrix <- confusionMatrix(testData$pred_class, testData$Churn, positive = "Yes")
print(conf_matrix)

# ROC Curve & AUC Score (AUC = 0.684)
roc_obj <- roc(
  response  = testData$Churn,
  predictor = testData$prob_Churn,
  levels    = c("No", "Yes"),
  direction = "<"
)
cat("Test Set AUC:", round(auc(roc_obj), 3), "\n")
plot(roc_obj, col = "#1f77b4", lwd = 2, main = "ROC Curve - HomeShop Churn Model")

# 5. Customer Persona Risk Simulation ------------------------------------------
personas <- data.frame(
  Profile             = c("Critical Risk", "Loyal VIP", "Moderate Risk", "Engaged Regular"),
  NumPurchases        = c(2, 20, 5, 8),
  SupportInteractions = c(9, 0, 4, 2),
  MembershipLevel     = c("Bronze", "Platinum", "Silver", "Gold"),
  Month               = c("Dec", "Jun", "Oct", "Jan"),
  Region              = c("Geneva", "Zurich", "Bern", "Lausanne")
)

personas$Predicted_Churn_Prob <- round(predict(model_refined, newdata = personas, type = "response"), 3)
print(personas)
