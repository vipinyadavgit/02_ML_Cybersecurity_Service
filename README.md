# Automotive Cybersecurity Service Adoption Prediction

## 🚗 Project Goal
> Build a machine learning solution that predicts which cybersecurity service package a vehicle is most likely to adopt.

This project is intended for automotive product, sales, and analytics teams who need a data-driven way to target cybersecurity subscriptions based on vehicle profile and connectivity capabilities.

---

## 🌟 Project Summary

This repository contains a complete ML workflow for automotive cybersecurity service adoption. It includes data ingestion, exploratory analysis, model training, evaluation, and business recommendations.

The final deliverable is a predictive model capable of classifying vehicles into one of five cybersecurity packages, plus reporting and plots to explain the model and its business value.

---

## 🎯 Business Problem

Modern vehicles are becoming increasingly connected and software-defined. As a result, cybersecurity packages must be tailored to vehicle features, connectivity, and subscription preferences.

The business problem is:
- Automotive teams currently lack a data-driven process to forecast the best cybersecurity package for each vehicle.
- This makes it hard to price, market, and recommend the right security tier.

The solution is a predictive model that classifies vehicle profiles into one of five cybersecurity service packages and provides interpretable insights for stakeholders.

---

## 📌 Key Objectives

- Develop a robust multi-class classification model
- Identify the vehicle features that most influence package adoption
- Generate actionable recommendations for product and sales teams
- Provide clear visualizations and business insights

---

## 📂 Project Structure

```
02_ML_Cybersecurity_Service/
│
├── dataset/
│   └── automotive_cybersecurity_adoption_dataset_3000.csv
│
├── docs/
│   ├── BRD_Automotive_Cybersecurity_Service_Adoption.docx
│   └── ... additional documentation ...
│
├── output_plots/
│   ├── 01_service_distribution.png
│   ├── 02_numerical_univariate_analysis.png
│   ├── 03_categorical_univariate_analysis.png
│   ├── 04_service_by_key_features.png
│   ├── 05_numerical_features_by_service.png
│   ├── 06_confusion_matrix.png
│   ├── 07_feature_importance.png
│   └── ... generated plot files ...
│
├── src/
│   └── automotive_cybersecurity_prediction.py
│
├── README.md
└── .venv/
```

---

## ▶️ How to Run

### 1. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run the main script

```powershell
python src/automotive_cybersecurity_prediction.py
```

### What happens next
- The script loads the dataset
- Performs EDA and feature engineering
- Trains and evaluates machine learning models
- Generates visual plots under `output_plots/`
- Prints metrics and business insight output to the terminal

---

## 🧠 What the Script Does

The workflow in `src/automotive_cybersecurity_prediction.py` is:

1. Load the dataset
2. Clean and preprocess data
3. Perform exploratory data analysis (EDA)
4. Encode categorical variables and scale numeric inputs
5. Train multiple classification models
6. Evaluate models with cross-validation and test data
7. Generate feature importance and business insights
8. Run sample predictions for example vehicles

---

## 📊 Models Trained & Compared

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline model |
| Decision Tree | Rule-based model for easy explanation |
| Random Forest | Ensemble model for strong performance and feature importance |

> The project chooses the best model based on weighted F1-score to handle imbalanced package adoption.

---

## 📦 Dataset Description

The dataset includes vehicle specifications, connectivity options, and the chosen cybersecurity service package.

### Main features
- `Brand`, `Model`, `Year`
- `Fuel_Type`, `Transmission`, `Price`, `Mileage`, `Engine_CC`, `Seating_Capacity`, `Vehicle_Segment`
- `ADAS_Level`, `Connected_Car`, `OTA_Update_Enabled`
- `Vehicle_Age`
- Target: `Cybersecurity_Service`

### Target packages
- `Basic Monitoring`
- `Threat Detection`
- `OTA Security`
- `Vulnerability Management`
- `Premium Cyber Suite`

---

## 📈 Output and Results

The project generates:
- Visual plots for service distribution and feature analysis
- Model performance metrics (accuracy, precision, recall, F1-score)
- A confusion matrix for the best model
- Feature importance rankings
- Business insights for product and sales strategy

---

## 📝 Documents in `docs/`

| Document | Description |
|---|---|
| `BRD_Automotive_Cybersecurity_Service_Adoption.docx` | Business requirements and project scope |
| `README.md` | Project summary and usage guide |
| `src/automotive_cybersecurity_prediction.py` | Main implementation script |

---

## 🛠 Tech Stack

| Package | Version |
|---|---|
| Python | 3.14.5 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |

**Install dependencies:**

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn python-docx
```

---

## 💡 Notes

- Keep the dataset file in `dataset/`.
- After running the script, review plots in `output_plots/`.
- The current implementation is designed for offline batch analysis, not real-time inference.
- You can extend the model with more vehicle brands, package types, or additional features.

---

## 🌱 Future Enhancements

Potential next steps:
- Add support for more cybersecurity package categories
- Integrate with dealership or CRM systems for live scoring
- Add explainability with SHAP or LIME
- Build a simple GUI or API for model predictions

---

## 📌 Final Thought
This project turns vehicle and cybersecurity data into actionable recommendations, helping stakeholders target the right cybersecurity subscription at the right time.
