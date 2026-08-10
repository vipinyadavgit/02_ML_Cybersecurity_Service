# =============================================================================
# Automotive Cybersecurity Service Prediction
# Problem Type : Multi-class Classification
# Target Variable : Cybersecurity_Service
# =============================================================================

# Import libraries
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# Folder to save plots only
os.makedirs("output_plots", exist_ok=True)


# Load Dataset
print("\nLOADING AUTOMOTIVE CYBERSECURITY DATASET")
df = pd.read_csv("../dataset/automotive_cybersecurity_adoption_dataset_3000.csv")

print("Dataset loaded Shape:", df.shape)


# EDA - Exploratory Data Analysis
# We look at the data to understand its structure before building any model

print("\n EDA")

# Basic checks: shape, types, first rows
print("\n--- Basic Info ---")
print("Shape:", df.shape)
df.info()
print(df.describe().T)
print(df.head())

# Check missing values and fill them with simple values
print("\n--- Null Value Check ---")
print(df.isnull().sum())

# Check and remove duplicate rows
print("\n--- Duplicate Check ---")
print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)


# Check for error values and convert them into missing values
error_vals = ['?', '@', '#', 'NA', 'N/A', 'none']

for col in df.columns:
    found = df[col].isin(error_vals).sum()

    if found > 0:
        print(col, ":", found, "error values found")
        df[col].replace(error_vals, np.nan, inplace=True)
        

# get list of text columns and number columns separately
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(exclude="object").columns.tolist()
print("\nText columns:", cat_cols)
print("\nNumber columns:", num_cols)


# Use mode for text columns and median for number columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\nCategorical columns:", cat_cols)
print("\nNumerical columns:", num_cols)

##  we are not doing IQR analysis or outlier removal because the dataset is synthetic and we want to keep all data points for model training because
##  In automotive data, a very high Price or Engine_CC may be a valid luxury vehicle—not an error.

# Target distribution plot
target_col = "Cybersecurity_Service"
print("\n--- Cybersecurity Service Distribution ---")
print(df[target_col].value_counts())


plt.figure(figsize=(9, 5))
df[target_col].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Cybersecurity Service Adoption")
plt.xlabel("Cybersecurity Service")
plt.ylabel("Number of Vehicles")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("output_plots/01_service_distribution.png", dpi=150)
plt.close()

# Univariate analysis: histograms for important numerical vehicle columns
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.hist(df["Price"], bins=20, color="steelblue", edgecolor="white")
plt.title("Price")

plt.subplot(2, 3, 2)
plt.hist(df["Mileage"], bins=20, color="steelblue", edgecolor="white")
plt.title("Mileage")

plt.subplot(2, 3, 3)
plt.hist(df["Engine_CC"], bins=20, color="steelblue", edgecolor="white")
plt.title("Engine_CC")

plt.subplot(2, 3, 4)
plt.hist(df["Year"], bins=10, color="steelblue", edgecolor="white")
plt.title("Year")

plt.subplot(2, 3, 5)
plt.hist(df["Vehicle_Age"], bins=10, color="steelblue", edgecolor="white")
plt.title("Vehicle_Age")

plt.subplot(2, 3, 6)
plt.hist(df["ADAS_Level"], bins=5, color="steelblue", edgecolor="white")
plt.title("ADAS_Level")

plt.suptitle("Univariate Analysis - Numerical Vehicle Features", fontsize=14)
plt.tight_layout()
plt.savefig("output_plots/02_numerical_univariate_analysis.png", dpi=150)
plt.close()

# Univariate analysis: bar charts for important categorical and binary columns
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
df["Brand"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Brand")
plt.xticks(rotation=25, ha="right")

plt.subplot(2, 3, 2)
df["Fuel_Type"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Fuel_Type")
plt.xticks(rotation=25, ha="right")

plt.subplot(2, 3, 3)
df["Transmission"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Transmission")
plt.xticks(rotation=25, ha="right")

plt.subplot(2, 3, 4)
df["Vehicle_Segment"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Vehicle_Segment")
plt.xticks(rotation=25, ha="right")

plt.subplot(2, 3, 5)
df["Connected_Car"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Connected_Car")
plt.xticks(rotation=0)

plt.subplot(2, 3, 6)
df["OTA_Update_Enabled"].value_counts().plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("OTA_Update_Enabled")
plt.xticks(rotation=0)

plt.suptitle("Univariate Analysis - Categorical and Binary Features", fontsize=14)
plt.tight_layout()
plt.savefig("output_plots/03_categorical_univariate_analysis.png", dpi=150)
plt.close()

# Bivariate analysis: categorical and binary features compared with service adoption
# first i use Seaborn countplot() which was giving simple raw counts but i want to show percentages,
# so i use crosstab() and stacked bar chartsn crosstab() counts combinations of two columns.
plt.figure(figsize=(15, 9))

plt.subplot(2, 2, 1)
table = pd.crosstab(df["Connected_Car"], df[target_col], normalize="index") * 100
table.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20") #ax=plt.gca()=get current axes
plt.title("Cybersecurity Service by Connected_Car")
plt.xlabel("Connected_Car")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(fontsize=7)

plt.subplot(2, 2, 2)
table = pd.crosstab(df["OTA_Update_Enabled"], df[target_col], normalize="index") * 100
table.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20")
plt.title("Cybersecurity Service by OTA_Update_Enabled")
plt.xlabel("OTA_Update_Enabled")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(fontsize=7)

plt.subplot(2, 2, 3)
table = pd.crosstab(df["Vehicle_Segment"], df[target_col], normalize="index") * 100
table.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20")
plt.title("Cybersecurity Service by Vehicle_Segment")
plt.xlabel("Vehicle_Segment")
plt.ylabel("Percentage")
plt.xticks(rotation=25, ha="right")
plt.legend(fontsize=7)

plt.subplot(2, 2, 4)
table = pd.crosstab(df["ADAS_Level"], df[target_col], normalize="index") * 100
table.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20")
plt.title("Cybersecurity Service by ADAS_Level")
plt.xlabel("ADAS_Level")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(fontsize=7)

plt.suptitle("Bivariate Analysis - Categorical Features vs Cybersecurity Service", fontsize=14)
plt.tight_layout()
plt.savefig("output_plots/04_service_by_key_features.png", dpi=150)
plt.close()

# Bivariate analysis: numerical features compared with cybersecurity service
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
sns.histplot(data=df, x="Price", hue=target_col, bins=20, element="step", stat="density", common_norm=False)
plt.title("Price vs Cybersecurity Service")

plt.subplot(2, 2, 2)
sns.histplot(data=df, x="Mileage", hue=target_col, bins=20, element="step", stat="density", common_norm=False)
plt.title("Mileage vs Cybersecurity Service")

plt.subplot(2, 2, 3)
sns.histplot(data=df, x="Engine_CC", hue=target_col, bins=20, element="step", stat="density", common_norm=False)
plt.title("Engine_CC vs Cybersecurity Service")

plt.subplot(2, 2, 4)
sns.histplot(data=df, x="Vehicle_Age", hue=target_col, bins=10, element="step", stat="density", common_norm=False)
plt.title("Vehicle_Age vs Cybersecurity Service")

plt.suptitle("Bivariate Analysis - Numerical Features vs Cybersecurity Service", fontsize=14)
plt.tight_layout()
plt.savefig("output_plots/05_numerical_features_by_service.png", dpi=150)
plt.close()


# FEATURE ENGINEERING
# Convert text values into numeric columns so ML models can understand them
print("\n FEATURE ENGINEERING")

# X contains the input columns; y is the target column
X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encoding creates one 0/1 column for every text category
X = pd.get_dummies(X, drop_first=True)

print("Shape after one-hot encoding:", X.shape)

# Encode the target package names as numbers
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print("Cybersecurity packages:", list(target_encoder.classes_))

# Split data: 80% for learning, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# StandardScaler: brings all features to the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#  MODEL BUILDING
# Train three simple models and compare them
print("\nMODEL BUILDING")

lr = LogisticRegression(max_iter=2000, random_state=42)
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

# 5-Fold Stratified Cross Validation
# Compare all 5 models using Accuracy and weighted multiclass Precision/Recall/F1
print("\n5-FOLD STRATIFIED CROSS VALIDATION")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

cv_models = {
    "Logistic Regression": (lr, X_train_scaled),
    "Decision Tree": (dt, X_train_scaled),
    "Random Forest": (rf, X_train_scaled)
}

cv_results = []
for name, (model, train_data) in cv_models.items():
    scores = cross_validate(model, train_data, y_train, cv=cv, scoring=scoring)
    cv_results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision_weighted"].mean(),
        "Recall": scores["test_recall_weighted"].mean(),
        "F1 Score": scores["test_f1_weighted"].mean()
    })
    print(name, "-> Average F1 Score:", round(scores["test_f1_weighted"].mean(), 4))

cv_df = pd.DataFrame(cv_results)
print("\nCross Validation Results (average of 5 folds):")
print(cv_df.round(4).to_string(index=False))

# Hyperparameter Tuning
# GridSearchCV tries every combination of parameters and picks the best one (by F1-score)
# cv=3 keeps it fast;
print("\nHYPERPARAMETER TUNING (GridSearchCV)")

# --- Logistic Regression ---
lr_params = {'C': [0.01, 0.1, 1, 10]}
lr_grid   = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'),
lr_params, cv=3, scoring='f1_weighted', n_jobs=1)
lr_grid.fit(X_train_scaled, y_train)
lr = lr_grid.best_estimator_
print("Logistic Regression best params:", lr_grid.best_params_)

# --- Decision Tree ---
dt_params = {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5, 10]}
dt_grid   = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'),
            dt_params, cv=3, scoring='f1_weighted', n_jobs=1)
dt_grid.fit(X_train_scaled, y_train)
dt = dt_grid.best_estimator_
print("Decision Tree best params:", dt_grid.best_params_)

# --- Random Forest ---
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10, 15]}
rf_grid   = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
            rf_params, cv=3, scoring='f1_weighted', n_jobs=1)
rf_grid.fit(X_train_scaled, y_train)
rf = rf_grid.best_estimator_
print("Random Forest best params:", rf_grid.best_params_)

print("\nAll models retrained with best hyperparameters.")

# update models dict so CV results reference the tuned models
models = {
    'Logistic Regression': lr,
    'Decision Tree'      : dt,
    'Random Forest'      : rf
}

# MODEL EVALUATION
# Evaluate each tuned model once using the unseen test data
print("\nMODEL EVALUATION")

y_pred_lr = lr.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)

print("\n--- Logistic Regression ---")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_lr, average="weighted", zero_division=0))
print("F1-Score :", f1_score(y_test, y_pred_lr, average="weighted", zero_division=0))

print("\n--- Decision Tree ---")
print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_dt, average="weighted", zero_division=0))
print("F1-Score :", f1_score(y_test, y_pred_dt, average="weighted", zero_division=0))

print("\n--- Random Forest ---")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_rf, average="weighted", zero_division=0))
print("F1-Score :", f1_score(y_test, y_pred_rf, average="weighted", zero_division=0))

# Model Comparison
# Put the final test scores in one table so selecting the best model is easy
print("\n--- Model Comparison Summary ---")
comparison = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr, average="weighted", zero_division=0),
        precision_score(y_test, y_pred_dt, average="weighted", zero_division=0),
        precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_dt, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_lr, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_dt, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    ]
}
comparison_df = pd.DataFrame(comparison).sort_values("F1-Score", ascending=False).reset_index(drop=True)
print(comparison_df.round(4).to_string(index=False))

# Choose the best model using its final weighted F1 score
best_model_name = comparison_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_test_data = X_test_scaled
best_prediction = best_model.predict(best_test_data)
print("\nBest Model:", best_model_name)
print("\nClassification Report:\n")
print(classification_report(y_test, best_prediction, target_names=target_encoder.classes_, zero_division=0))

# Confusion matrix plot explains which packages are predicted correctly
plt.figure(figsize=(9, 7))
cm = confusion_matrix(y_test, best_prediction)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted Service")
plt.ylabel("Actual Service")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("output_plots/06_confusion_matrix.png", dpi=150)
plt.close()


# FEATURE IMPORTANCE
# Random Forest can show which vehicle features influenced its decisions most
print("\nFEATURE IMPORTANCE")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print(feature_importance.head(15).round(4).to_string(index=False))

plt.figure(figsize=(10, 7))
sns.barplot(data=feature_importance.head(15), x="Importance", y="Feature", color="teal")
plt.title("Top 15 Important Features - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Vehicle Feature")
plt.tight_layout()
plt.savefig("output_plots/07_feature_importance.png", dpi=150)
plt.close()

print("\nBusiness interpretation:")
print("The top features are the vehicle details that most help the model identify")
print("which cybersecurity package a customer is likely to choose.")
print("Features such as price, ADAS level, connected-car capability and OTA updates")
print("are especially useful because they describe how digital and advanced a vehicle is.")


# PREDICTION EXAMPLES
# Make predictions for two sample vehicles and show every package probability
print("\nPREDICTION EXAMPLES")

def predict_vehicle(name, vehicle):
    sample = pd.DataFrame([vehicle])

    # Make the new vehicle columns match the columns used during training
    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=X.columns, fill_value=0)

    if best_model_name == "Logistic Regression":
        sample_for_model = scaler.transform(sample)
    else:
        sample_for_model = sample

    probabilities = best_model.predict_proba(sample_for_model)[0]
    predicted_number = best_model.predict(sample_for_model)[0]
    predicted_service = target_encoder.inverse_transform([predicted_number])[0]

    print("\n" + name)
    print("Predicted Cybersecurity Service:", predicted_service)
    print("Probabilities:")
    for service, probability in zip(target_encoder.classes_, probabilities):
        print(f"  {service}: {probability:.2%}")

vehicle_a = {
    "Brand": "Hyundai", "Model": "Creta", "Year": 2023,
    "Fuel_Type": "Hybrid", "Transmission": "Automatic",
    "Price": 2200000, "Mileage": 18, "Engine_CC": 1500,
    "Seating_Capacity": 5, "Vehicle_Segment": "Premium",
    "ADAS_Level": 2, "Connected_Car": "Yes",
    "OTA_Update_Enabled": "Yes", "Vehicle_Age": 3
}

vehicle_b = {
    "Brand": "Tata", "Model": "Nexon", "Year": 2023,
    "Fuel_Type": "Petrol", "Transmission": "Manual",
    "Price": 850000, "Mileage": 18, "Engine_CC": 1200,
    "Seating_Capacity": 5, "Vehicle_Segment": "Budget",
    "ADAS_Level": 0, "Connected_Car": "No",
    "OTA_Update_Enabled": "No", "Vehicle_Age": 3
}

predict_vehicle("Example Vehicle A", vehicle_a)
predict_vehicle("Example Vehicle B", vehicle_b)


# BUSINESS INSIGHTS
print("\nBUSINESS INSIGHTS")

# Find the most common cybersecurity service package
service_counts = df[target_col].value_counts()
most_common_package = service_counts.index[0]

# Find the top brands for Premium Cyber Suite adoption
premium_by_brand = pd.crosstab(df["Brand"], df[target_col], normalize="index")
premium_by_brand = premium_by_brand * 100
premium_by_brand = premium_by_brand.round(1)

if "Premium Cyber Suite" in premium_by_brand.columns:
    premium_brand_answers = premium_by_brand["Premium Cyber Suite"]
    premium_brand_answers = premium_brand_answers.sort_values(ascending=False)
    premium_brand_answers = premium_brand_answers.head(10)
else:
    premium_brand_answers = pd.Series(dtype=float)

# Find service mix by Connected_Car, OTA, and ADAS level
connected_car_mix = pd.crosstab(df["Connected_Car"], df[target_col], normalize="index")
connected_car_mix = connected_car_mix * 100
connected_car_mix = connected_car_mix.round(1)

ota_mix = pd.crosstab(df["OTA_Update_Enabled"], df[target_col], normalize="index")
ota_mix = ota_mix * 100
ota_mix = ota_mix.round(1)

adas_mix = pd.crosstab(df["ADAS_Level"], df[target_col], normalize="index")
adas_mix = adas_mix * 100
adas_mix = adas_mix.round(1)

# Find advanced package adoption by vehicle segment
advanced_services = ["OTA Security", "Vulnerability Management", "Premium Cyber Suite"]
is_advanced_service = df[target_col].isin(advanced_services)
segment_data = df.copy()
segment_data["Advanced_Service"] = is_advanced_service
advanced_segment_adoption = segment_data.groupby("Vehicle_Segment")["Advanced_Service"].mean()
advanced_segment_adoption = advanced_segment_adoption * 100
advanced_segment_adoption = advanced_segment_adoption.round(1)
advanced_segment_adoption = advanced_segment_adoption.sort_values(ascending=False)

print("1. Most commonly adopted package:", most_common_package)

print("\n2. Top brands by Premium Cyber Suite adoption (%):")
if not premium_brand_answers.empty:
    print(premium_brand_answers.to_string())
else:
    print("Premium Cyber Suite is not present in the current dataset.")

print("\n3. Service mix by Connected_Car (%):")
print(connected_car_mix.to_string())

print("\n4. Service mix by OTA_Update_Enabled (%):")
print(ota_mix.to_string())

print("\n5. Service mix by ADAS_Level (%):")
print(adas_mix.to_string())

print("\n6. Advanced package adoption by vehicle segment (%):")
print(advanced_segment_adoption.to_string())

print("\n7. Recommendation: offer advanced packages to connected, OTA-enabled and high-ADAS vehicles.")
print("Use the vehicle segment with the highest advanced adoption as the first sales priority.")
print("Note: package price is not available, so this is not an actual revenue calculation.")

print("\nCompleted")
