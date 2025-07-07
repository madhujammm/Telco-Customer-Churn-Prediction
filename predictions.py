import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/Telco-Customer-Churn.csv')
df.head()

df.shape

df.info()

df.isnull().sum()

df.duplicated().sum()

df.head(10)

df.Churn.value_counts().plot(kind='bar')
#THe target column is imbalanced

df.columns

"""FEATURE ENG."""

df = df.drop(['customerID'],axis=1)
df.head()

df.gender.unique()

df.gender = df.gender.replace(['Male','Female'],[1,0])
df.head()

df.Partner.unique()

df.Partner = df.Partner.replace(['Yes','No'],[1,0])
df.head()

df.Dependents.unique()

df.Dependents = df.Dependents.replace(['No','Yes'],[0,1])
df.head()

df.PhoneService.unique()

df.PhoneService = df.PhoneService.replace(['No','Yes'],[0,1])
df.head()

df.PaperlessBilling.unique()

df.PaperlessBilling = df.PaperlessBilling.replace(['Yes','No'],[1,0])
df.head()

df.Churn.unique()

df.Churn = df.Churn.replace(['No','Yes'],[0,1])
df.head()

df.MultipleLines.unique()

df.MultipleLines = df.MultipleLines.replace(['No phone service', 'No', 'Yes'],[0,1,2])
df.head()

df.InternetService.unique()

df.InternetService = df.InternetService.replace(['DSL', 'Fiber optic', 'No'],[1,2,0])
df.head()

df.OnlineSecurity.unique()

df.OnlineSecurity = df.OnlineSecurity.replace(['No','Yes','No internet service'],[1,2,0])
df.head()

df.OnlineBackup.unique()

df.OnlineBackup = df.OnlineBackup.replace(['Yes', 'No', 'No internet service'],[2,1,0])
df.head()

df.DeviceProtection.unique()

df.DeviceProtection = df.DeviceProtection.replace(['No', 'Yes', 'No internet service'],[1,2,0])
df.head()

df.TechSupport.unique()

df.TechSupport = df.TechSupport.replace(['No', 'Yes', 'No internet service'],[1,2,0])
df.head()

df.StreamingTV.unique()

df.StreamingTV = df.StreamingTV.replace(['No', 'Yes', 'No internet service'],[1,2,0])
df.head()

df.StreamingMovies.unique()

df.StreamingMovies = df.StreamingMovies.replace(['No', 'Yes', 'No internet service'],[1,2,0])
df.head()

df.Contract.unique()

df.Contract = df.Contract.replace(['Month-to-month', 'One year', 'Two year'],[0,1,2])
df.head()

df.PaperlessBilling.unique()

df.PaperlessBilling = df.PaperlessBilling.replace(['Yes','No'],[1,0])
df.head()

df.PaymentMethod.unique()

df.PaymentMethod = df.PaymentMethod.replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'],[1,2,3,4])
df.head()

df.info()

# Step 1: Convert everything to string first (so str methods won't crash)
df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip()

# Step 2: Replace empty strings with NaN
df['TotalCharges'].replace('', np.nan, inplace=True)

# Step 3: Convert to float (non-numeric values will turn into NaN automatically)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Step 4: Drop rows with NaN in TotalCharges
df.dropna(subset=['TotalCharges'], inplace=True)

df.info()

df = df.dropna()
df.isnull().sum() # returned all zeros so good to go[type float64, no missing values]

"""EDA"""

#Now check the heatmap
corr = df.corr()
plt.figure(figsize=(25,15))
sns.heatmap(corr,annot=True,cbar=True,cmap='plasma')
plt.show()

# Define number of columns for the subplot grid
num_cols = 2
num_rows = -(-len(df.columns) // num_cols)  # Ceiling division to get required rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))  # Adjust size dynamically
axes = axes.flatten()  # Flatten to easily iterate

for i, col in enumerate(df.columns):
    sns.distplot(df[col], ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # Ensure proper spacing
plt.show()

#Checking the distributions
#check dist of all columns
for col in df.columns:
  plt.figure(figsize=(10,6))
  sns.distplot(df[col])
  plt.show()

"""Data is imbalaned so now balanching it [balancing technique]"""

#THIS IS A TECHNIQUE FOR OVERSAMPLING


#the data in target column is imbalanced. So we will now oversample it
#Just apply oversampling in target column

from sklearn.utils import resample
#create two different dataframe of majority and minority class
df_majority = df[(df['Churn']==0)]
df_minority = df[(df['Churn']==1)]
# upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,    # sample with replacement
                                 n_samples= 5163, # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
df = pd.concat([df_minority_upsampled, df_majority])

df['Churn'].value_counts()

#Splitting the dataset]
x = df.drop(['Churn'],axis=1)
y = df.Churn

#Apply the train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train_scaled = ss.fit_transform(x_train)
x_test_scaled = ss.transform(x_test)

#model selections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier

#objects
lr = LogisticRegression()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
xgb = XGBClassifier()
svc = SVC()
knn = KNeighborsClassifier()
nb = GaussianNB()
lgb = LGBMClassifier()

#Fittings
lr.fit(x_train_scaled,y_train)
rf.fit(x_train_scaled,y_train)

gb.fit(x_train_scaled,y_train)
xgb.fit(x_train_scaled,y_train)

svc.fit(x_train_scaled,y_train)
knn.fit(x_train_scaled,y_train)

nb.fit(x_train_scaled,y_train)
lgb.set_params(verbosity=-1)
lgb.fit(x_train_scaled,y_train)

#preds
lrpred = lr.predict(x_test_scaled)
rfpred = rf.predict(x_test_scaled)
gbpred = gb.predict(x_test_scaled)
xgbpred = xgb.predict(x_test_scaled)
svcpred = svc.predict(x_test_scaled)
knnpred = knn.predict(x_test_scaled)
nbpred = nb.predict(x_test_scaled)
lgbpred = lgb.predict(x_test_scaled)

#Evaluations
from sklearn.metrics import accuracy_score
lracc = accuracy_score(y_test,lrpred)
rfacc = accuracy_score(y_test,rfpred)
gbacc = accuracy_score(y_test,gbpred)
xgbacc = accuracy_score(y_test,xgbpred)
svcacc = accuracy_score(y_test,svcpred)
knnacc = accuracy_score(y_test,knnpred)
nbacc = accuracy_score(y_test,nbpred)
lgbacc = accuracy_score(y_test,lgbpred)

print('LOGISTIC REG',lracc)
print('RANDOM FOREST',rfacc)
print('GB',gbacc)
print('XGB',xgbacc)
print('SVC',svcacc)
print('KNN',knnacc)
print('NB',nbacc)
print('LIGHT GBM',lgbacc)

"""confusion matrix heatmap"""

#NOW CHECK THE CONFUSION MATRIX(for specific model)
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test,rfpred) #Enter the model pred here
plt.title('Heatmap of Confusion matrix',fontsize=15)
sns.heatmap(cm,annot=True)
plt.show()

#classification report
#NOW we will check the classification report
print(classification_report(y_test,rfpred))

#cross val score
#(TO CHECK IF THE MODEL HAS OVERFITTED OR UNDERFITTED)

from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(estimator=rf,X=x_train_scaled,y=y_train)
print('Cross Val Acc Score of RANDOM FOREST model is ---> ',cross_val)
print('\n Cross Val Mean Acc Score of RANDOM FOREST model is ---> ',cross_val.mean())

import shap

# Train best model (Gradient Boosting)
best_model = rf.fit(x_train_scaled, y_train)

# SHAP analysis
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(x_test_scaled)



residuals = y_test - best_model.predict(x_test_scaled)

# Residual vs Predicted plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=best_model.predict(x_test_scaled), y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")

# Q-Q plot for normality check
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt);

from sklearn.metrics import mean_squared_error

# Convert RMSE to dollar terms (assuming prices are in $1,000s)
rmse_dollars = np.sqrt(mean_squared_error(y_test, best_model.predict(x_test_scaled))) * 1000
print(f"Average Prediction Error: ${rmse_dollars:,.2f}")

# Compare to median house price
median_price = np.median(y_train) * 1000
print(f"Error as % of Median Price: {rmse_dollars/median_price:.2%}")

from sklearn.model_selection import cross_val_predict

# Get cross-val predictions with uncertainty
predictions = cross_val_predict(best_model, x_train_scaled, y_train, cv=5, method="predict")

# Plot actual vs predicted with 95% CI
sns.regplot(x=y_train, y=predictions)
plt.title("Cross-Validated Predictions")

#NOW save the model
import pickle

#Saving the model
pickle.dump(rf,open('Custormer_Churn_RF.pickle','wb'))

#loading the model
Custormer_Churn_RF_model = pickle.load(open('Custormer_Churn_RF.pickle','rb'))

#Predicting the output
y_pred = Custormer_Churn_RF_model.predict(x_test_scaled)

#confusion matrix
print('Confusion matrix of Custormer_Churn_RF : \n', confusion_matrix(y_test,y_pred),'\n')

#showing off the accuracy score
print('Accuracy Score on testing data is',accuracy_score(y_test,y_pred))

