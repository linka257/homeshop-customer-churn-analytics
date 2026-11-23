# End-to-End Customer Churn Analytics: Power BI & R

**Author:** Linka Ndizeye ([GitHub](https://github.com) | [LinkedIn](https://linkedin.com))  
**Tech Stack:** Power BI, R (`caret`, `pROC`, `car`, `ranger`, `tidyverse`), Logistic Regression & Random Forest  

---

## Business Problem & Data Overview
HomeShop, a mid-sized Swiss online electronics retailer, experienced rising attrition (**29.9% baseline churn rate**) across **24,017 customer records**. This two-phase project pairs **diagnostic visual exploration (Power BI)** with **predictive risk scoring and threshold optimization (R)** to allocate retention budgets efficiently.

> **Data Availability Note:** The original `Homeshop.csv` dataset was provided as part of university coursework at EU Business School Geneva and is not redistributed publicly. `HomeShop_Churn_Model.R` automatically generates a structurally matching synthetic dataset (`n = 24,017`) when run standalone so any reviewer can reproduce the full pipeline.

### Data Dictionary

| Variable | Type | Description |
| :--- | :--- | :--- |
| `Churn` | Binary (`0/1`) | Target variable (`1` = Churned, `0` = Retained; 29.9% positive class) |
| `NumPurchases` | Numeric | Total completed orders |
| `TotalSpending` | Numeric | Cumulative spend in CHF |
| `WebsiteVisits` | Numeric | Total platform sessions |
| `SupportInteractions` | Numeric | Customer service tickets/calls logged |
| `MembershipLevel` | Categorical | Loyalty tier (`Bronze`, `Silver`, `Gold`, `Platinum`) |
| `MembershipMonths` | Numeric | Account tenure in months |
| `DiscountRate` | Numeric | Tier discount rate (collinear with `MembershipLevel`, excluded) |
| `Age` / `AgeBand` | Numeric / Cat. | Customer age grouped into `<30`, `30–44`, and `45+` bands |
| `Month` | Categorical | Cohort / observation month (`Apr` set as reference baseline) |
| `Region` | Categorical | Swiss city/canton (`Basel`, `Bern`, `Geneva`, `Lausanne`, `Zurich`) |
| `Gender` / `IncomeGroup` | Categorical | Customer demographic attributes |

---

## Phase 1: Diagnostic Analysis (Power BI)
Exploratory visual analysis across all candidate variables established key empirical patterns:
* **Loyalty Tier Gradient:** Bronze members churned at **40%**, falling to **27%** (Silver), **20%** (Gold), and **9%** (Platinum).
* **Support Friction Escalation:** Customers with 0–2 support tickets churned at 13–31%, rising progressively from 3+ tickets and peaking sharply at **6–8+ interactions**.
* **Non-Linear Age Effect:** Churn spiked among younger (`23–29`) and older (`45–51`) shoppers while remaining lowest among core shoppers (`30–44`)—motivating categorical `AgeBand` encoding rather than a raw linear term in Phase 2.
* **Weak Standalone Demographics:** Gender (`29%` vs. `31%`) and Swiss canton (`29–31%`) showed minimal standalone variance, though regional and monthly fixed effects were retained in Phase 2 to control for seasonal/local confounding.

### Diagnostic Dashboards
![Power BI Behavioral & Engagement Analysis](powerbi_page1.png)

![Power BI Demographics & Loyalty Analysis](powerbi_page2.png)

---

## Phase 2: Predictive Modeling & Validation (R)

### 1. Modeling Rigor & Benchmark Comparison
* **Collinearity Handling:** `DiscountRate` was excluded due to exact linear dependency (singularity) with `MembershipLevel`; Variance Inflation Factors (`car::vif`) confirmed low multicollinearity across remaining predictors.
* **Cross-Validation & Benchmark:** Evaluated via **10-fold cross-validation (`caret::train`)**, binary logistic regression achieved an out-of-sample **ROC-AUC of 0.684 (95% CI: 0.668–0.700)**—matching a **Random Forest (`ranger`)** benchmark while retaining direct odds-ratio interpretability.
* **Calibration & Lift:** Decile calibration confirmed strong probability alignment across risk tiers, with the **top risk decile capturing ~2.1x the baseline churn rate**—making the model well suited for **risk ranking** even with modest overall discrimination.

### 2. Key Drivers (Odds Ratios vs. Reference Baselines)
* **Support Interactions (`OR = 1.33`):** Each additional support contact increases the **odds of churn by 33.2%** ($p < 0.001$).
* **Loyalty Tier (`OR = 0.20` to `0.59` vs. `Bronze`):** Platinum members have **80.2% lower odds** of churning than Bronze members ($p < 0.001$); Gold and Silver show **63.7%** and **41.4% lower odds**, respectively.
* **Purchase Frequency (`OR = 0.92`):** Each additional purchase reduces churn odds by **8.1%** ($p < 0.001$).
* **Seasonality (`OR = 1.37` vs. `April` baseline):** December cohorts exhibit **36.6% higher churn odds** ($p < 0.001$), alongside a modest regional effect in Geneva (`OR = 1.14`, $p = 0.011$).

### 3. Classification Threshold & Cost-Benefit Trade-Off
At the default `0.50` probability cutoff, overall accuracy (**70.7%**) sits only slightly above the **70.1% No-Information Rate** (predicting all customers retained), yielding **94.7% specificity** but only **14.6% sensitivity** (catching ~1 in 7 churners). Because retention outreach (automated email/voucher at ~CHF 5/customer) is far cheaper than losing a customer, `HomeShop_Churn_Model.R` sweeps thresholds via `pROC::coords`:
* **`0.50` Cutoff:** High precision (`54.0%`), low recall (`14.6%`) — misses most at-risk buyers.
* **Youden-Optimal / `~0.30` Cutoff:** Substantially increases churner capture (recall) while preserving positive expected net retention value.

---

## Business Recommendations & Limitations
1. **Early Escalation at 3+ Support Tickets (Preventing the 6–8 Ticket Cliff):** While churn peaks at 6–8+ interactions, intervening at **3 tickets** acts as an early-warning circuit breaker before dissatisfaction becomes irreversible.
2. **Threshold-Driven Retention Campaigns (`p >= 0.30`):** Score active customers weekly and trigger automated re-engagement offers for accounts exceeding a `0.30` predicted churn probability.
3. **Behavioral Nudges Toward Higher Tiers:** Use targeted bonus-point campaigns to encourage repeat orders among Bronze/Silver users. *(Note: Because higher tiers and lower support friction correlate with organic customer satisfaction rather than purely causing retention, incentives should be A/B tested before full rollout.)*

### Limitations
* **Observational Data & Endogeneity:** Support tickets and tier status reflect customer experience and engagement; causal lift requires randomized A/B testing.
* **Feature Scope:** Incorporating product-category returns, delivery lead times, and time-since-last-purchase (recency) would likely improve AUC beyond `0.684`.
