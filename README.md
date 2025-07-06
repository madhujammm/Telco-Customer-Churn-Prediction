# 📊 Telco Customer Churn Prediction

This project predicts whether a customer will **churn** (i.e., leave the company) based on their demographic, service usage, and account data using **machine learning models (Random Forest, Decision Tree)**.
It also involves **EDA, feature engineering, hyperparameter tuning, and model saving** for deployment.

---

## 📁 Table of Contents

* [📌 Problem Statement](#-problem-statement)
* [📦 Dataset](#-dataset)
* [🛠️ Tech Stack](#️-tech-stack)
* [🔍 EDA Insights](#-eda-insights)
* [⚙️ Project Workflow](#️-project-workflow)
* [🚀 Getting Started](#-getting-started)
* [📈 Model Evaluation](#-model-evaluation)
* [💾 Model Saving & Loading](#-model-saving--loading)
* [📂 Directory Structure](#-directory-structure)

---

## 📌 Problem Statement

A telecom company wants to identify customers who are likely to **churn** so that targeted actions can be taken to retain them.

---

## 📦 Dataset

* **Name**: Telco-Customer-Churn.csv
* **Rows**: 7043
* **Columns**: 21
* **Target column**: `Churn` (Yes/No)

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn (DecisionTree, RandomForest, GridSearchCV, StandardScaler)
* Joblib (for model saving)

---

## 🔍 EDA Insights

| Feature                  | Insight                                  |
| ------------------------ | ---------------------------------------- |
| `Contract_One year`      | Long-term contracts reduce churn         |
| `tenure`, `TotalCharges` | Low tenure & low spending → higher churn |
| `TechSupport_Yes`        | Access to support reduces churn          |
| `PaymentMethod`          | Electronic checks linked to higher churn |

---

## ⚙️ Project Workflow

### 1. Data Preprocessing

* Loaded dataset using `pandas`
* Converted `TotalCharges` to numeric
* Dropped rows with missing values
* Binary encoded columns like `gender`, `Partner`, `Churn`
* One-hot encoded categorical variables
* Feature scaling with `StandardScaler` on:

  * `tenure`, `MonthlyCharges`, `TotalCharges`

### 2. Exploratory Data Analysis (EDA)

* Count plots and histograms
* Correlation heatmaps
* Identified key churn-driving features

### 3. Model Training

* **Models used**:

  * `DecisionTreeClassifier`
  * `RandomForestClassifier`
* **Split**: `train_test_split` (80/20)

### 4. Hyperparameter Tuning

* Used `GridSearchCV` on Random Forest
* Parameters tuned:

  * `n_estimators`, `max_depth`, `min_samples_split`
* Best Parameters:

  ```python
  {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 150}
  ```

### 5. Model Evaluation

* Accuracy:

  * Decision Tree: `~78.96%`
  * Random Forest (default): `~79.31%`
  * Random Forest (tuned): `~80.39%` (CV)

### 6. Model Exporting

* Saved models using `joblib`:

  ```python
  joblib.dump(best_model, 'best_random_forest_model.pkl')
  ```

---

## 🚀 Getting Started

### 🔧 Requirements

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### ▶️ Run the Code

1. Clone the repo
2. Open `churn_prediction.ipynb` or run the script in your IDE
3. Load the dataset: `Telco-Customer-Churn.csv`
4. Run cells for EDA, preprocessing, model training, and evaluation

---

## 📈 Model Evaluation

| Model                    | Accuracy        |
| ------------------------ | --------------- |
| Decision Tree            | 78.96%          |
| Random Forest (default)  | 79.31%          |
| 🔧 Random Forest (tuned) | **80.39% (CV)** |

---

## 💾 Model Saving & Loading

```python
# Save
joblib.dump(grid_rf.best_estimator_, 'best_random_forest_model.pkl')

# Load
model = joblib.load('best_random_forest_model.pkl')

# Predict
predictions = model.predict(X_test)
```

---

## 📂 Directory Structure

```
📦 Telco-Churn-Prediction/
├── Telco-Customer-Churn.csv
├── churn_prediction.ipynb
├── best_random_forest_model.pkl
├── decision_tree_model.pkl
├── README.md
```

---


