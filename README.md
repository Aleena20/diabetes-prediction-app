# 🩺 Diabetes Prediction — Beginner Data Science Project

A beginner-friendly end-to-end machine learning project that predicts whether a patient is diabetic based on diagnostic health measurements.

Inspired by the structure of classic beginner ML projects like Titanic Survival Prediction.

---

## 📂 Project Structure

```
diabetes_prediction/
├── diabetes.csv                  # Dataset (Pima Indians Diabetes)
├── diabetes_prediction.py        # Main ML pipeline script
├── requirements.txt              # Python dependencies
├── eda_distributions.png         # Feature histograms by outcome
├── correlation_heatmap.png       # Feature correlation matrix
├── outcome_distribution.png      # Class balance pie chart
├── confusion_matrices.png        # Confusion matrices for all models
├── roc_curves.png                # ROC curves for all models
└── feature_importance.png        # Random Forest feature importances
```

---

## 📊 Dataset

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

## 🔬 Steps in the Pipeline

1. **Data Loading & Overview** — shape, dtypes, missing values, class balance
2. **Exploratory Data Analysis (EDA)** — histograms, heatmaps, outcome distribution
3. **Preprocessing** — handle biologically invalid zeros, median imputation, train/test split, feature scaling
4. **Model Training** — Logistic Regression, Random Forest, K-Nearest Neighbours
5. **Evaluation** — Accuracy, ROC-AUC, 5-fold Cross-Validation, Confusion Matrices, ROC Curves
6. **Feature Importance** — Random Forest importances
7. **Model Comparison** — summary table + best model selection

---

## 🚀 Getting Started

### 1. Clone this project
```bash
git clone <your-repo-url>
cd diabetes_prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python diabetes_prediction.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 📈 Results (Example)

| Model | Test Accuracy | ROC-AUC | CV Accuracy |
|---|---|---|---|
| Logistic Regression | ~0.78 | ~0.84 | ~0.77 |
| Random Forest | ~0.80 | ~0.87 | ~0.79 |
| K-Nearest Neighbours | ~0.74 | ~0.80 | ~0.74 |

> ✅ **Random Forest** typically performs best on this dataset.

---

## 💡 Key Learnings

- Handling missing data encoded as zeros (common in medical datasets)
- Importance of stratified train/test splitting for imbalanced classes
- Why ROC-AUC is a better metric than accuracy for medical diagnosis tasks
- Feature scaling is crucial for distance-based models (KNN) and regularised models (Logistic Regression)

---

## 🔧 Possible Improvements

- Hyperparameter tuning with GridSearchCV
- SMOTE for class imbalance handling
- XGBoost / LightGBM models
- SHAP values for model explainability
- Deploy with a Streamlit app

---

*Built as part of a beginner data science portfolio.*
