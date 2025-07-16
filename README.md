# 📊 Customer Churn Prediction - Telecom Sector 

## 📌 Problem Statement

A telecom company is experiencing customer churn and wants to identify which customers are likely to leave their services. By predicting churn in advance, the company can take targeted retention actions to reduce losses.

---

## 🧠 Objectives

- Identify features that impact customer churn
- Handle imbalanced dataset with resampling
- Train multiple classification models
- Evaluate performance using accuracy, confusion matrix, and cross-validation
- Interpret results using SHAP
- Save the best model for future predictions

---

## 📂 Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7043
- **Columns**: 20
- **Target**: `Churn` (0: No, 1: Yes)

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: 
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`, `lightgbm`, `shap`
- **Model Persistence**: `pickle`

---

## 📈 Workflow

### 🔹 1. Data Preprocessing
- Removed customer ID
- Converted categorical values to numerical
- Cleaned `TotalCharges` column (object → float)
- Dropped missing values

### 🔹 2. EDA (Exploratory Data Analysis)
- Correlation heatmaps
- Distribution plots for each feature
- Class imbalance check (`Churn` was imbalanced)

### 🔹 3. Data Balancing
- Oversampled the minority class using `resample()` (up to 5163 rows)

### 🔹 4. Modeling
Trained and evaluated 8 models:
- Logistic Regression
- Random Forest ✅
- Gradient Boosting
- XGBoost
- SVC
- KNN
- Naive Bayes
- LightGBM

### 🔹 5. Model Evaluation
- Accuracy score
- Confusion matrix
- Classification report
- Cross-validation (5 folds)
- SHAP analysis for interpretability

---

## ✅ Best Model: Random Forest

| Metric           | Value         |
|------------------|---------------|
| Accuracy         | **90.9%**     |
| Precision (1)    | 0.88          |
| Recall (1)       | 0.96          |
| F1-Score (1)     | 0.92          |

---

## 🔍 SHAP & Interpretability

- SHAP used to explain feature importance and understand model predictions.
- Residual and Q-Q plots used for visual validation.

---

## 💾 Model Saving & Prediction

- Best model (`RandomForestClassifier`) saved using `pickle`
- Can be loaded and used for predictions on new data

---

## 📌 How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the notebook or Python script
python churn_model.py

# Step 3: Load saved model and predict
pickle.load(open('Custormer_Churn_RF.pickle', 'rb'))
