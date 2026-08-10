# =============================================================================
# HR Employee Attrition Prediction - Capstone Project
# Problem Type : Binary Classification
# Target Variable : Attrition (Yes / No)
# =============================================================================


# Import Libraries 
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report
)

warnings.filterwarnings('ignore')

# folder to save all plots
os.makedirs("output_plots", exist_ok=True)


# Load Dataset
print(" LOADING DATASET")

df = pd.read_csv("../dataset/HR-Employee-Attrition.csv")
print("Dataset loaded. Shape:", df.shape)


# STEP 3: EDA - Exploratory Data Analysis
# We look at the data to understand its structure before building any model
print("\n EDA")

# Basic checks: shape, types, first rows
print("\n--- Basic Info ---")
print("Shape:", df.shape)
df.info()
print(df.describe().T)
print(df.head())

# Drop columns that have the same value in every row (useless for model)
print("\n--- Dropping useless columns ---")
df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
print("Remaining columns:", df.shape[1])

# Check for null/missing values
print("\n--- Null Value Check ---")
print(df.isnull().sum())

# Check for duplicate rows
print("\n--- Duplicate Check ---")
print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# get list of text columns and number columns separately
text_cols   = df.select_dtypes(include="object").columns.tolist()
number_cols = df.select_dtypes(exclude="object").columns.tolist()
print("\nText columns:", text_cols)
print("\nNumber columns:", number_cols)


# for categorical(text) columns: Use mode
for col in text_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# for discrete number columns: use median 
for col in number_cols:
    df[col].fillna(df[col].median(), inplace=True)



# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(exclude="object").columns.tolist()
print("\nCategorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# Check for error values like ?, @, etc.
print("\n--- Error Value Check ---")
error_vals = ['?', '@', '#', 'NA', 'N/A', 'none']
for col in cat_cols:
    found = df[col].isin(error_vals).sum()
    if found > 0:
        print(col, ":", found, "error values found, replacing...")
        df[col].replace(error_vals, np.nan, inplace=True)
        df[col].fillna(df[col].mode()[0], inplace=True)


# Outlier detection using IQR method
print("\n--- Outlier Check (IQR Method) ---")
for col in num_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    LB    = Q1 - 1.5 * IQR   # lower boundary
    UB    = Q3 + 1.5 * IQR   # upper boundary
    count = df[(df[col] < LB) | (df[col] > UB)].shape[0]
    if count > 0:
        print("  " + col + ":", count, "outliers")

# boxplot for the 4 columns mentioned in the Final Capstone Project document
# these are also the columns with the most outliers in the dataset
plt.figure(figsize=(16, 6))

plt.subplot(1, 4, 1)
plt.boxplot(df['Age'])
plt.title('Age')

plt.subplot(1, 4, 2)
plt.boxplot(df['MonthlyIncome'])
plt.title('MonthlyIncome')

plt.subplot(1, 4, 3)
plt.boxplot(df['TotalWorkingYears'])
plt.title('TotalWorkingYears')

plt.subplot(1, 4, 4)
plt.boxplot(df['YearsAtCompany'])
plt.title('YearsAtCompany')

plt.suptitle("Boxplots - Outlier Check")
plt.tight_layout()
plt.savefig("output_plots/01_outlier_boxplots.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] 01_outlier_boxplots.png")

# Target 
print("\n--- Target Variable: Attrition ---")
print(df['Attrition'].value_counts())


plt.figure(figsize=(6, 4))
df['Attrition'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'], edgecolor='black')
plt.title("Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output_plots/02_target_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] 02_target_distribution.png")

# Univariate analysis: histograms for numerical columns
plt.figure(figsize=(16, 12))

plt.subplot(3, 3, 1)
plt.hist(df['Age'], bins=20, color='steelblue', edgecolor='white')
plt.title('Age')

plt.subplot(3, 3, 2)
plt.hist(df['MonthlyIncome'], bins=20, color='steelblue', edgecolor='white')
plt.title('MonthlyIncome')

plt.subplot(3, 3, 3)
plt.hist(df['TotalWorkingYears'], bins=20, color='steelblue', edgecolor='white')
plt.title('TotalWorkingYears')

plt.subplot(3, 3, 4)
plt.hist(df['YearsAtCompany'], bins=20, color='steelblue', edgecolor='white')
plt.title('YearsAtCompany')

plt.subplot(3, 3, 5)
plt.hist(df['DistanceFromHome'], bins=20, color='steelblue', edgecolor='white')
plt.title('DistanceFromHome')

plt.subplot(3, 3, 6)
plt.hist(df['NumCompaniesWorked'], bins=20, color='steelblue', edgecolor='white')
plt.title('NumCompaniesWorked')

plt.subplot(3, 3, 7)
plt.hist(df['PercentSalaryHike'], bins=20, color='steelblue', edgecolor='white')
plt.title('PercentSalaryHike')

plt.subplot(3, 3, 8)
plt.hist(df['TrainingTimesLastYear'], bins=10, color='steelblue', edgecolor='white')
plt.title('TrainingTimesLastYear')

plt.subplot(3, 3, 9)
plt.hist(df['YearsSinceLastPromotion'], bins=15, color='steelblue', edgecolor='white')
plt.title('YearsSinceLastPromotion')

plt.suptitle("Univariate Analysis - Numerical Features", fontsize=14)
plt.tight_layout()
plt.savefig("output_plots/03_univariate_numerical.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] 03_univariate_numerical.png")

# bar charts for categorical columns
plt.figure(figsize=(16, 9))

plt.subplot(2, 3, 1)
df['Department'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Department')
plt.xticks(rotation=30)

plt.subplot(2, 3, 2)
df['JobRole'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('JobRole')
plt.xticks(rotation=30)

plt.subplot(2, 3, 3)
df['MaritalStatus'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('MaritalStatus')
plt.xticks(rotation=0)

plt.subplot(2, 3, 4)
df['BusinessTravel'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('BusinessTravel')
plt.xticks(rotation=20)

plt.subplot(2, 3, 5)
df['Gender'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Gender')
plt.xticks(rotation=0)

plt.subplot(2, 3, 6)
df['OverTime'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('OverTime')
plt.xticks(rotation=0)

plt.suptitle("Univariate Analysis - Categorical Features", fontsize=14)
plt.tight_layout()
plt.savefig("output_plots/04_univariate_categorical.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] 04_univariate_categorical.png")

# Bivariate analysis: how each feature relates to Attrition
# This tells us WHY employees are leaving - useful for business recommendations
print("\n--- Bivariate Analysis (Feature vs Attrition) ---")

no_attr  = df[df['Attrition'] == 'No']
yes_attr = df[df['Attrition'] == 'Yes']

# attrition rate by categorical features
plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
ct = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack().fillna(0) * 100
ct.plot(kind='bar', ax=plt.gca(), color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Attrition by Department')
plt.ylabel('%')
plt.xticks(rotation=20)

plt.subplot(2, 3, 2)
ct = df.groupby('MaritalStatus')['Attrition'].value_counts(normalize=True).unstack().fillna(0) * 100
ct.plot(kind='bar', ax=plt.gca(), color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Attrition by MaritalStatus')
plt.ylabel('%')
plt.xticks(rotation=0)

plt.subplot(2, 3, 3)
ct = df.groupby('BusinessTravel')['Attrition'].value_counts(normalize=True).unstack().fillna(0) * 100
ct.plot(kind='bar', ax=plt.gca(), color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Attrition by BusinessTravel')
plt.ylabel('%')
plt.xticks(rotation=20)

plt.subplot(2, 3, 4)
ct = df.groupby('Gender')['Attrition'].value_counts(normalize=True).unstack().fillna(0) * 100
ct.plot(kind='bar', ax=plt.gca(), color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Attrition by Gender')
plt.ylabel('%')
plt.xticks(rotation=0)

plt.subplot(2, 3, 5)
ct = df.groupby('OverTime')['Attrition'].value_counts(normalize=True).unstack().fillna(0) * 100
ct.plot(kind='bar', ax=plt.gca(), color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Attrition by OverTime')
plt.ylabel('%')
plt.xticks(rotation=0)

plt.subplot(2, 3, 6)
ct = df.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack().fillna(0) * 100
ct.plot(kind='bar', ax=plt.gca(), color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Attrition by JobRole')
plt.ylabel('%')
plt.xticks(rotation=30)

plt.title("Bivariate Analysis - Attrition % by Categorical Features", fontsize=14)
plt.tight_layout()
plt.savefig("output_plots/05_bivariate_categorical_vs_attrition.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] 05_bivariate_categorical_vs_attrition.png")

# Histogram:-  numerical features vs Attrition
plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.hist(no_attr['Age'],  bins=20, color='steelblue', alpha=0.6, edgecolor='white', label='No')
plt.hist(yes_attr['Age'], bins=20, color='tomato',    alpha=0.6, edgecolor='white', label='Yes')
plt.title('Age vs Attrition')
plt.legend()

plt.subplot(2, 3, 2)
plt.hist(no_attr['MonthlyIncome'],  bins=20, color='steelblue', alpha=0.6, edgecolor='white', label='No')
plt.hist(yes_attr['MonthlyIncome'], bins=20, color='tomato',    alpha=0.6, edgecolor='white', label='Yes')
plt.title('MonthlyIncome vs Attrition')
plt.legend()

plt.subplot(2, 3, 3)
plt.hist(no_attr['TotalWorkingYears'],  bins=20, color='steelblue', alpha=0.6, edgecolor='white', label='No')
plt.hist(yes_attr['TotalWorkingYears'], bins=20, color='tomato',    alpha=0.6, edgecolor='white', label='Yes')
plt.title('TotalWorkingYears vs Attrition')
plt.legend()

plt.subplot(2, 3, 4)
plt.hist(no_attr['YearsAtCompany'],  bins=20, color='steelblue', alpha=0.6, edgecolor='white', label='No')
plt.hist(yes_attr['YearsAtCompany'], bins=20, color='tomato',    alpha=0.6, edgecolor='white', label='Yes')
plt.title('YearsAtCompany vs Attrition')
plt.legend()

plt.subplot(2, 3, 5)
plt.hist(no_attr['DistanceFromHome'],  bins=20, color='steelblue', alpha=0.6, edgecolor='white', label='No')
plt.hist(yes_attr['DistanceFromHome'], bins=20, color='tomato',    alpha=0.6, edgecolor='white', label='Yes')
plt.title('DistanceFromHome vs Attrition')
plt.legend()

plt.subplot(2, 3, 6)
plt.hist(no_attr['YearsSinceLastPromotion'],  bins=15, color='steelblue', alpha=0.6, edgecolor='white', label='No')
plt.hist(yes_attr['YearsSinceLastPromotion'], bins=15, color='tomato',    alpha=0.6, edgecolor='white', label='Yes')
plt.title('YearsSinceLastPromotion vs Attrition')
plt.legend()

plt.suptitle("Bivariate Analysis - Numerical Features vs Attrition", fontsize=14)
plt.tight_layout()
plt.savefig("output_plots/06_bivariate_numerical_vs_attrition.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] 06_bivariate_numerical_vs_attrition.png")

# correlation heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap - Numerical Features")
plt.tight_layout()
plt.savefig("output_plots/07_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] 07_correlation_heatmap.png")



# Feature Engineering
# Convert text columns to numbers so ML model can understand them

print("\nS FEATURE ENGINEERING")

# work on a copy so original df is not changed
df_model = df.copy()

# encoding target: Yes = 1, No = 0
df_model['Attrition'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})
print("Attrition encoded: Yes=1, No=0")

# encoding binary columns using map 
df_model['Gender']   = df_model['Gender'].map({'Male': 1, 'Female': 0})
df_model['OverTime'] = df_model['OverTime'].map({'Yes': 1, 'No': 0})
print("Gender and OverTime encoded")

# ordinal encoding for BusinessTravel (has a natural order: no travel , rarely , frequently)
df_model['BusinessTravel'] = df_model['BusinessTravel'].map(
    {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
)
print("BusinessTravel encoded as ordinal")

# one-hot encoding for remaining nominal columns (no natural order between categories)
# drop_first=True avoids the dummy variable trap / multicollinearity
df_model = pd.get_dummies(df_model, columns=['Department', 'EducationField', 'JobRole', 'MaritalStatus'], drop_first=True)
print("One-hot encoding done. Shape:", df_model.shape)

# X = input features, y = output 
X = df_model.drop(columns=['Attrition'])
y = df_model['Attrition']

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:\n", y.value_counts())

# 80% for training, 20% for testing
# stratify=y keeps the same Yes/No ratio in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)

# StandardScaler: brings all features to the same scale
# We fit only on train data to avoid data leakage into test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Scaling done.")


# Model Building
# Training 5 different classification models one by one
# class_weight='balanced' handles class imbalance automatically

print("\nSTEP 5: MODEL BUILDING")

# Model 1: Logistic Regression
# Good simple baseline model for binary classification problems
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
print("Logistic Regression trained")

# Model 2: Decision Tree
# Easy to explain to non-technical stakeholders; shows decision rules visually
dt = DecisionTreeClassifier(max_depth=20, class_weight='balanced')
dt.fit(X_train_scaled, y_train)
print("Decision Tree trained")

# Model 3: Random Forest
# Combines many decision trees; usually gives best accuracy + feature importance
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')
rf.fit(X_train_scaled, y_train)
print("Random Forest trained")

# Model 4: KNN (K-Nearest Neighbors)
# Classifies based on similarity to nearest training examples
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
print("KNN trained")

# Model 5: Naive Bayes
# Fast probabilistic model; assumes all features are independent of each other
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
print("Naive Bayes trained")


# 5-Fold Stratified Cross Validation

print("\n5-FOLD STRATIFIED CROSS VALIDATION")

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1']

models = {
    'Logistic Regression': lr,
    'Decision Tree'      : dt,
    'Random Forest'      : rf,
    'KNN'                : knn,
    'Naive Bayes'        : nb
}

cv_results = []
for name, model in models.items():
    scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring)
    cv_results.append({
        'Model'    : name,
        'Accuracy' : scores['test_accuracy'].mean(),
        'Precision': scores['test_precision'].mean(),
        'Recall'   : scores['test_recall'].mean(),
        'F1-Score' : scores['test_f1'].mean()
    })
    print(name, "-> F1:", round(scores['test_f1'].mean(), 4))

cv_df = pd.DataFrame(cv_results).sort_values('F1-Score', ascending=False).reset_index(drop=True)
print("\nCross Validation Results (averaged over 5 folds):")
print(cv_df.round(4))


# Hyperparameter Tuning
# GridSearchCV tries every combination of parameters and picks the best one (by F1-score)
# cv=3 keeps it fast; scoring='f1' is preferred for imbalanced classes
print("\n HYPERPARAMETER TUNING (GridSearchCV)")

# --- Logistic Regression ---
lr_params = {'C': [0.01, 0.1, 1, 10]}
lr_grid   = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'),
lr_params, cv=3, scoring='f1', n_jobs=1)
lr_grid.fit(X_train_scaled, y_train)
lr = lr_grid.best_estimator_
print("Logistic Regression best params:", lr_grid.best_params_)

# --- Decision Tree ---
dt_params = {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5, 10]}
dt_grid   = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'),
            dt_params, cv=3, scoring='f1', n_jobs=1)
dt_grid.fit(X_train_scaled, y_train)
dt = dt_grid.best_estimator_
print("Decision Tree best params:", dt_grid.best_params_)

# --- Random Forest ---
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10, 15]}
rf_grid   = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
            rf_params, cv=3, scoring='f1', n_jobs=1)
rf_grid.fit(X_train_scaled, y_train)
rf = rf_grid.best_estimator_
print("Random Forest best params:", rf_grid.best_params_)

# --- KNN ---
knn_params = {'n_neighbors': [3, 5, 7, 9, 11]}
knn_grid   = GridSearchCV(KNeighborsClassifier(),
            knn_params, cv=3, scoring='f1', n_jobs=1)
knn_grid.fit(X_train_scaled, y_train)
knn = knn_grid.best_estimator_
print("KNN best params:", knn_grid.best_params_)

# --- Naive Bayes ---
nb_params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
nb_grid   = GridSearchCV(GaussianNB(), nb_params, cv=3, scoring='f1', n_jobs=1)
nb_grid.fit(X_train_scaled, y_train)
nb = nb_grid.best_estimator_
print("Naive Bayes best params:", nb_grid.best_params_)

print("\nAll models retrained with best hyperparameters.")

# update models dict so CV results reference the tuned models
models = {
    'Logistic Regression': lr,
    'Decision Tree'      : dt,
    'Random Forest'      : rf,
    'KNN'                : knn,
    'Naive Bayes'        : nb
}


# Model Evaluation
# Compare all 5 models using Accuracy, Precision, Recall, F1-Score
# F1-Score is the most important metric here because data is imbalanced
# =============================================================================
print("\nMODEL EVALUATION")

y_pred_lr  = lr.predict(X_test_scaled)
y_pred_dt  = dt.predict(X_test_scaled)
y_pred_rf  = rf.predict(X_test_scaled)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_nb  = nb.predict(X_test_scaled)

#  Logistic Regression 
print("\n--- Logistic Regression ---")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall   :", recall_score(y_test, y_pred_lr))
print("F1-Score :", f1_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=['No Attrition', 'Attrition']))

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix:\n", cm_lr)

# --- Decision Tree ---
print("\n--- Decision Tree ---")
print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall   :", recall_score(y_test, y_pred_dt))
print("F1-Score :", f1_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, target_names=['No Attrition', 'Attrition']))

cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:\n", cm_dt)


# --- Random Forest ---
print("\n--- Random Forest ---")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1-Score :", f1_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=['No Attrition', 'Attrition']))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:\n", cm_rf)


# --- KNN ---
print("\n--- KNN ---")
print("Accuracy :", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn))
print("Recall   :", recall_score(y_test, y_pred_knn))
print("F1-Score :", f1_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, target_names=['No Attrition', 'Attrition']))

cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix:\n", cm_knn)


# --- Naive Bayes ---
print("\n--- Naive Bayes ---")
print("Accuracy :", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb))
print("Recall   :", recall_score(y_test, y_pred_nb))
print("F1-Score :", f1_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, target_names=['No Attrition', 'Attrition']))

cm_nb = confusion_matrix(y_test, y_pred_nb)
print("Confusion Matrix:\n", cm_nb)


# --- Model Comparison ---
print("\n--- Model Comparison Summary ---")
comparison = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'Naive Bayes'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_nb)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_nb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_nb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_nb)
    ]
}

comparison_df = pd.DataFrame(comparison)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
print(comparison_df.round(4))


best_row = comparison_df.iloc[0]
print("\nBest Model (by F1-Score):", best_row['Model'])
print("F1-Score :", round(best_row['F1-Score'], 4))
print("Accuracy :", round(best_row['Accuracy'], 4))


# Feature Importance (using Random Forest)

print("\nFEATURE IMPORTANCE (Random Forest)")

feature_names = X.columns.tolist()

importance_scores = rf.feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature'   : feature_names,
    'Importance': importance_scores
})

feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

print("\nTop Features that influence Attrition the most:")
print(feat_imp_df.round(4))


#  Predict for New / Sample Employees

print("\n PREDICT FOR SAMPLE EMPLOYEES")

# taking 2 sample employees from the test set for demonstration
sample_X      = X_test_scaled[:2]
sample_actual = y_test.iloc[:2].values

# use model which won the comparison (highest F1-Score)
model_map = {
    'Logistic Regression': lr,
    'Decision Tree'      : dt,
    'Random Forest'      : rf,
    'KNN'                : knn,
    'Naive Bayes'        : nb
}
best_model      = model_map[best_row['Model']]
sample_pred     = best_model.predict(sample_X)
sample_probability    = best_model.predict_proba(sample_X)[:, 1]

print("\nPredictions using best model:", best_row['Model'])
print("Employee 1:")
print("  Actual    :", "Attrition" if sample_actual[0] == 1 else "No Attrition")
print("  Predicted :", "Attrition" if sample_pred[0] == 1   else "No Attrition")
print("  Probability:", round(sample_probability[0] * 100, 2), "%")

print("Employee 2:")
print("  Actual    :", "Attrition" if sample_actual[1] == 1 else "No Attrition")
print("  Predicted :", "Attrition" if sample_pred[1] == 1   else "No Attrition")
print("  Probability:", round(sample_probability[1] * 100, 2), "%")

print("\nAll plots saved to:", os.path.abspath("output_plots"))


print("PROJECT COMPLETE")


