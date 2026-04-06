# Diabetes Prediction — Beginner Data Science Project

A beginner-friendly end-to-end machine learning project with an interactive Streamlit web app that predicts whether a patient is diabetic based on diagnostic health measurements.

Inspired by the structure of classic beginner ML projects like Titanic Survival Prediction.

---

## Live Demo

> Deploy to Streamlit Community Cloud (free) — see deployment instructions below.

---
## Screenshots

| Prediction Tab | EDA Tab | Model Comparison |
|---|---|---|
| Patient sliders + probability bar | Feature distributions + heatmap | ROC curves + confusion matrices |

---
## Features

- **Interactive Prediction** — adjust 8 clinical sliders and get an instant diabetic/non-diabetic prediction with probability scores
- **3 Models** — Logistic Regression, Random Forest, K-Nearest Neighbours (switchable from sidebar)
- **EDA Tab** — per-feature distribution histograms, box plots, and a correlation heatmap
- **Model Comparison Tab** — ROC curves, confusion matrices, feature importances, and full classification reports

---
## Project Structure

```
diabetes_prediction/
├── app.py                        # Streamlit application
├── diabetes.csv                  # Dataset (Pima Indians Diabetes)
├── diabetes_prediction.py        # Main ML pipeline script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── eda_distributions.png         # Feature histograms by outcome
├── correlation_heatmap.png       # Feature correlation matrix
├── outcome_distribution.png      # Class balance pie chart
├── confusion_matrices.png        # Confusion matrices for all models
├── roc_curves.png                # ROC curves for all models
├── feature_importance.png        # Random Forest feature importances
└── README.md                     # This file
 
```

---

## Dataset

**Pima Indians Diabetes Dataset**
- 768 patient records
- 8 input features, 1 binary target
- Source: Originally from the National Institute of Diabetes and Digestive and Kidney Diseases

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2hr oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic risk score) |
| Age | Age in years |
| **Outcome** | **1 = Diabetic, 0 = Non-Diabetic** |

---

## Steps in the Pipeline

1. **Data Loading & Overview** — shape, dtypes, missing values, class balance
2. **Exploratory Data Analysis (EDA)** — histograms, heatmaps, outcome distribution
3. **Preprocessing** — handle biologically invalid zeros, median imputation, train/test split, feature scaling
4. **Model Training** — Logistic Regression, Random Forest, K-Nearest Neighbours
5. **Evaluation** — Accuracy, ROC-AUC, 5-fold Cross-Validation, Confusion Matrices, ROC Curves
6. **Feature Importance** — Random Forest importances
7. **Model Comparison** — summary table + best model selection
8. **Deploy** - to Streamlit Community Cloud (Free)
---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/diabetes-prediction-app.git
cd diabetes-prediction-app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Results (Example)

| Model | Test Accuracy | ROC-AUC | CV Accuracy (5-fold)|
|---|---|---|---|
| Logistic Regression | ~0.78 | ~0.84 | ~0.77 |
| Random Forest | ~0.80 | ~0.87 | ~0.79 |
| K-Nearest Neighbours | ~0.74 | ~0.80 | ~0.74 |

> **Random Forest** typically performs best on this dataset.

---

## Key Learnings

- Handling missing data encoded as zeros (common in medical datasets)
- Importance of stratified train/test splitting for imbalanced classes
- Why ROC-AUC is a better metric than accuracy for medical diagnosis tasks
- Feature scaling is crucial for distance-based models (KNN) and regularised models (Logistic Regression)

---

## Possible Improvements

- Hyperparameter tuning with GridSearchCV
- SMOTE for class imbalance handling
- XGBoost / LightGBM models
- SHAP values for model explainability



