# ==============================================================================
# Project: Customer Churn Prediction & Retention Analytics at HomeShop
# Author:  Linka Ndizeye (https://github.com/linka257)
# Tools:   R (caret, pROC, car, ranger, tidyverse) | Paired with Power BI
# Dataset: 24,017 retail customer records across Swiss regions
# ==============================================================================

library(caret)
library(pROC)
library(car)
library(ranger)
library(tidyverse)

set.seed(123)

# 1. Load Data (with Synthetic Fallback for Reproducibility) -------------------
if (file.exists("Homeshop.csv")) {
  Homeshop <- read_csv("Homeshop.csv", show_col_types = FALSE)
} else {
  message("Course CSV not found locally. Generating synthetic dataset (n = 24,017) with matching structure...")
  n <- 24017
  Homeshop <- tibble(
    NumPurchases        = rpois(n, lambda = 6),
    TotalSpending       = round(runif(n, 50, 5000), 2),
    WebsiteVisits       = rpois(n, lambda = 15),
    SupportInteractions = rpois(n, lambda = 2.2),
    MembershipMonths    = sample(1:60, n, replace = TRUE),
    MembershipLevel     = sample(c("Bronze", "Silver", "Gold", "Platinum"), n, replace = TRUE, prob = c(0.45, 0.30, 0.18, 0.07)),
    Month               = sample(c("Apr", "Aug", "Dec", "Feb", "Jan", "Jul", "Jun", "Mai", "Mar", "Nov", "Oct", "Sep"), n, replace = TRUE),
    Gender              = sample(c("Female", "Male"), n, replace = TRUE),
    Age                 = sample(20:55, n, replace = TRUE),
    Region              = sample(c("Basel", "Bern", "Geneva", "Lausanne", "Zurich"), n, replace = TRUE),
    IncomeGroup         = sample(c("Low", "Medium", "High"), n, replace = TRUE)
  ) %>%
    mutate(
      # DiscountRate is deterministically mapped to MembershipLevel (exact collinearity)
      DiscountRate = case_when(
        MembershipLevel == "Bronze"   ~ 0.02,
        MembershipLevel == "Silver"   ~ 0.05,
        MembershipLevel == "Gold"     ~ 0.10,
        MembershipLevel == "Platinum" ~ 0.15
      ),
      lin_pred = -0.40 - 0.084 * NumPurchases + 0.286 * SupportInteractions -
        ifelse(MembershipLevel == "Silver", 0.53,
               ifelse(MembershipLevel == "Gold", 1.01,
                      ifelse(MembershipLevel == "Platinum", 1.62, 0))) +
        ifelse(Age < 30 | Age >= 45, 0.35, 0) +
        ifelse(Month == "Dec", 0.31, 0),
      Churn = rbinom(n, 1, plogis(lin_pred))
    ) %>%
    select(-lin_pred)
}

# 2. Feature Engineering & Factor Encoding -------------------------------------
Homeshop <- Homeshop %>%
  drop_na() %>%
  mutate(
    Churn = factor(Churn, levels = c(0, 1), labels = c("No", "Yes")),
    # Capture the non-linear U-shaped Age pattern observed in Power BI Phase 1
    AgeBand = factor(
      case_when(
        Age < 30 ~ "Young (<30)",
        Age <= 44 ~ "Core (30-44)",
        TRUE ~ "Mature (45+)"
      ),
      levels = c("Core (30-44)", "Young (<30)", "Mature (45+)")
    ),
    # Set explicit reference levels so interpretation is unambiguous
    MembershipLevel = relevel(factor(MembershipLevel), ref = "Bronze"),
    Month           = relevel(factor(Month), ref = "Apr"),
    Region          = relevel(factor(Region), ref = "Basel")
  )

cat("Class Balance (No Information Rate = 70.1% Retained):\n")
print(round(prop.table(table(Homeshop$Churn)), 3))

# 3. Train / Test Split (80% Train, 20% Test) ----------------------------------
trainIndex <- createDataPartition(Homeshop$Churn, p = 0.8, list = FALSE)
trainData  <- Homeshop[trainIndex, ]
testData   <- Homeshop[-trainIndex, ]

# 4. Collinearity Diagnostics & Model Specification ----------------------------
# Note: DiscountRate is excluded because it is deterministically assigned by
# MembershipLevel (exact linear dependency / singularity in glm).
model_check <- glm(
  Churn ~ NumPurchases + TotalSpending + WebsiteVisits + SupportInteractions +
    MembershipMonths + MembershipLevel + AgeBand + Month + Region + Gender + IncomeGroup,
  data = trainData,
  family = binomial
)

cat("\nVariance Inflation Factors (GVIF):\n")
print(vif(model_check))

# Refined specification combining behavioral, loyalty, age-band, and temporal/regional controls
model_formula <- Churn ~ NumPurchases + SupportInteractions + MembershipLevel + AgeBand + Month + Region

# 5. 10-Fold Cross-Validation & Benchmark Comparison ---------------------------
cv_ctrl <- trainControl(
  method          = "cv",
  number          = 10,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Model A: Penalized / Standard Logistic Regression
cv_glm <- train(
  model_formula,
  data      = trainData,
  method    = "glm",
  family    = binomial,
  metric    = "ROC",
  trControl = cv_ctrl
)

# Model B: Random Forest Benchmark (ranger)
cv_rf <- train(
  model_formula,
  data       = trainData,
  method     = "ranger",
  metric     = "ROC",
  trControl  = cv_ctrl,
  num.trees  = 200,
  tuneGrid   = expand.grid(mtry = 3, splitrule = "gini", min.node.size = 20)
)

cat("\n10-Fold CV ROC Comparison (Logistic Regression vs. Random Forest):\n")
print(resamples(list(LogisticRegression = cv_glm, RandomForest = cv_rf)) %>% summary())

# Fit final interpretable logistic model on full training set
final_glm <- glm(model_formula, data = trainData, family = binomial)
summary(final_glm)

# Odds Ratios with 95% Confidence Intervals
or_table <- exp(cbind(OR = coef(final_glm), confint.default(final_glm)))
cat("\nOdds Ratios & 95% CIs:\n")
print(round(or_table, 3))

# 6. Out-of-Sample Evaluation, Calibration & Lift ------------------------------
testData$prob_Churn <- predict(final_glm, newdata = testData, type = "response")

# ROC Curve & 95% Confidence Interval
roc_obj <- roc(response = testData$Churn, predictor = testData$prob_Churn, levels = c("No", "Yes"), direction = "<")
auc_ci  <- ci.auc(roc_obj)
cat(sprintf("\nTest Set AUC: %.3f (95%% CI: %.3f - %.3f)\n", auc_ci[2], auc_ci[1], auc_ci[3]))

# Default 0.50 Cutoff Confusion Matrix (illustrating sensitivity/specificity imbalance)
pred_50 <- factor(ifelse(testData$prob_Churn >= 0.50, "Yes", "No"), levels = c("Yes", "No"))
cm_50   <- confusionMatrix(pred_50, factor(testData$Churn, levels = c("Yes", "No")), positive = "Yes")
print(cm_50)

# Calibration Check by Predicted Risk Decile
calibration_df <- testData %>%
  mutate(decile = ntile(prob_Churn, 10)) %>%
  group_by(decile) %>%
  summarise(
    n             = n(),
    mean_pred     = round(mean(prob_Churn), 3),
    observed_rate = round(mean(Churn == "Yes"), 3),
    lift          = round(observed_rate / mean(testData$Churn == "Yes"), 2)
  )
cat("\nCalibration & Decile Lift Table (Top Decile = Highest Risk):\n")
print(calibration_df)

# 7. Threshold Sweep & Cost-Benefit Optimization -------------------------------
# Simple retention economics:
# - Automated email / voucher campaign cost per targeted customer: CHF 5
# - Net margin value of retaining an at-risk customer: CHF 80 (assuming 25% save rate -> CHF 20 expected value)
cost_contact <- 5
value_save   <- 80 * 0.25  # CHF 20 expected gross value per true churner reached

youden_coords <- coords(roc_obj, x = "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity", "precision"))
thresholds    <- c(0.50, round(youden_coords$threshold, 3), 0.30, 0.25)

threshold_eval <- map_dfr(thresholds, function(t) {
  preds <- ifelse(testData$prob_Churn >= t, "Yes", "No")
  tp <- sum(preds == "Yes" & testData$Churn == "Yes")
  fp <- sum(preds == "Yes" & testData$Churn == "No")
  fn <- sum(preds == "No"  & testData$Churn == "Yes")
  targeted <- tp + fp
  net_benefit <- (tp * value_save) - (targeted * cost_contact)
  tibble(
    Threshold     = t,
    Sensitivity   = round(tp / (tp + fn), 3),
    Precision     = round(tp / targeted, 3),
    Customers_Hit = targeted,
    Net_Value_CHF = round(net_benefit, 0)
  )
})
cat("\nThreshold Sweep & Retention Campaign Economics:\n")
print(threshold_eval)

# 8. Persona Risk Simulation ---------------------------------------------------
personas <- data.frame(
  Profile             = c("Critical Risk", "Loyal VIP", "Moderate Risk", "Engaged Regular"),
  NumPurchases        = c(2, 20, 5, 8),
  SupportInteractions = c(9, 0, 4, 2),
  MembershipLevel     = factor(c("Bronze", "Platinum", "Silver", "Gold"), levels = levels(trainData$MembershipLevel)),
  AgeBand             = factor(c("Young (<30)", "Core (30-44)", "Core (30-44)", "Core (30-44)"), levels = levels(trainData$AgeBand)),
  Month               = factor(c("Dec", "Jun", "Oct", "Jan"), levels = levels(trainData$Month)),
  Region              = factor(c("Geneva", "Zurich", "Bern", "Lausanne"), levels = levels(trainData$Region))
)
personas$Predicted_Churn_Prob <- round(predict(final_glm, newdata = personas, type = "response"), 3)
cat("\nSimulated Persona Churn Probabilities:\n")
print(personas)
