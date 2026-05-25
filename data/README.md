# data/

This folder contains the raw dataset used for all analysis and modelling.

## File

| File | Description |
|---|---|
| `churn_2024.csv` | Raw telecom customer dataset — 10,000 rows × 19 columns (18 features + 1 target) |

---

## Dataset Overview

- **Source:** Provided as part of the Interpretable Data Science group assignment
- **Rows:** 10,000 customers
- **Columns:** 19 (18 features + `churn` target)
- **Missing values:** None — the dataset is complete
- **Class balance:** ~27% churned (`churn = 1`), ~73% retained (`churn = 0`)

---

## Data Dictionary

### Target Variable

| Column | Type | Description |
|---|---|---|
| `churn` | Binary (0/1) | Whether the customer churned (1) or was retained (0) |

### Account & Demographics

| Column | Type | Description |
|---|---|---|
| `state` | Categorical | U.S. state of the customer |
| `account_length` | Numeric | Number of months the customer has been with the company |

### Plan Subscriptions (Binary)

| Column | Type | Description |
|---|---|---|
| `international_plan` | Binary (0/1) | Whether the customer has an international calling plan |
| `voice_mail_plan` | Binary (0/1) | Whether the customer has a voicemail plan |
| `wifi_bundle` | Binary (0/1) | Whether the customer has a WiFi bundle |
| `phone_bundle` | Binary (0/1) | Whether the customer has a phone bundle |

### Usage

| Column | Type | Description |
|---|---|---|
| `total_usage` | Numeric | Total usage (proxy for call volume / data consumption) |
| `total_intl_calls` | Numeric | Number of international calls placed |
| `ad_spend` | Numeric | Marketing ad spend attributed to this customer *(dropped: VIF > 100, derived from total_usage)* |

### Service Quality

| Column | Type | Description |
|---|---|---|
| `number_service_outage` | Numeric | Number of service outages the customer experienced |
| `service_outage_reported` | Numeric | Number of outages formally reported *(dropped: VIF > 50,000, near-duplicate of above)* |

### Customer Service

| Column | Type | Description |
|---|---|---|
| `number_customer_service_calls` | Numeric | Number of calls to customer service |
| `number_customer_service_interactions` | Numeric | Total customer service interactions (superset of calls) *(dropped: VIF ~ 5)* |

### Financial

| Column | Type | Description |
|---|---|---|
| `discount` | Numeric | Discount rate applied to the customer's plan |

---

## Preprocessing Applied in the Notebook

| Step | Detail |
|---|---|
| Dropped features | `service_outage_reported`, `ad_spend`, `number_customer_service_interactions` (severe VIF) |
| Engineered features | `calls_per_month`, `usage_intensity`, `frustration_index` |
| Encoding | One-hot encoding on `state` with `drop_first=True` |
| Split | 60% train / 20% validation / 20% test, stratified on `churn` |
| Scaling | `StandardScaler` fit on training set only |

> **Do not modify this file.** The notebook reads it as-is and applies all transformations programmatically.
