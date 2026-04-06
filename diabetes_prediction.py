# -*- coding: utf-8 -*-
"""
Diabetes Survival Prediction
=============================
Dataset : Pima Indians Diabetes Dataset (768 patients, 8 features)
Target  : Outcome — 1 = Diabetic, 0 = Non-Diabetic
Models  : Logistic Regression, Random Forest, K-Nearest Neighbours
Author  : <Your Name>
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("diabetes.csv")

print("=" * 55)
print("DIABETES PREDICTION — DATA OVERVIEW")
print("=" * 55)
print(f"\nShape : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nOutcome distribution:\n{df['Outcome'].value_counts()}")
print(f"\nBasic stats:\n{df.describe().round(2)}")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Feature Distributions by Outcome", fontsize=16, fontweight="bold")

features = df.columns[:-1]
for ax, col in zip(axes.flatten(), features):
    df[df["Outcome"] == 0][col].hist(ax=ax, alpha=0.6, color="steelblue", label="Non-Diabetic", bins=20)
    df[df["Outcome"] == 1][col].hist(ax=ax, alpha=0.6, color="tomato", label="Diabetic", bins=20)
    ax.set_title(col)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] eda_distributions.png")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] correlation_heatmap.png")

# Outcome pie chart
plt.figure(figsize=(6, 6))
df["Outcome"].value_counts().plot.pie(
    labels=["Non-Diabetic", "Diabetic"],
    autopct="%1.1f%%",
    colors=["steelblue", "tomato"],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
plt.title("Outcome Distribution", fontsize=14, fontweight="bold")
plt.ylabel("")
plt.tight_layout()
plt.savefig("outcome_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] outcome_distribution.png")

# ─────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────
# Columns where 0 is physiologically impossible → treat as missing
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train / test split (80 / 20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
}

results = {}

print("\n" + "=" * 55)
print("MODEL RESULTS")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cv   = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy").mean()

    results[name] = {"Accuracy": acc, "ROC-AUC": auc, "CV Accuracy": cv, "model": model, "y_prob": y_prob}

    print(f"\n{'-'*40}")
    print(f"  {name}")
    print(f"{'-'*40}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  ROC-AUC Score : {auc:.4f}")
    print(f"  CV Accuracy   : {cv:.4f} (5-fold)")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))

# ─────────────────────────────────────────────
# 6. CONFUSION MATRICES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    model  = res["model"]
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-Diabetic", "Diabetic"],
                yticklabels=["Non-Diabetic", "Diabetic"])
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] confusion_matrices.png")

# ─────────────────────────────────────────────
# 7. ROC CURVES
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 6))
colors = ["steelblue", "tomato", "seagreen"]

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['ROC-AUC']:.3f})", color=color, lw=2)

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — All Models", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] roc_curves.png")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
importances.plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("Feature Importances — Random Forest", fontsize=14, fontweight="bold")
plt.ylabel("Importance Score")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] feature_importance.png")

# ─────────────────────────────────────────────
# 9. MODEL COMPARISON SUMMARY
# ─────────────────────────────────────────────
summary = pd.DataFrame({
    name: {"Test Accuracy": res["Accuracy"], "ROC-AUC": res["ROC-AUC"], "CV Accuracy": res["CV Accuracy"]}
    for name, res in results.items()
}).T.round(4)

print("\n" + "=" * 55)
print("MODEL COMPARISON SUMMARY")
print("=" * 55)
print(summary.to_string())

best_model = summary["ROC-AUC"].idxmax()
print(f"\n[BEST] Best model by ROC-AUC: {best_model} ({summary.loc[best_model, 'ROC-AUC']:.4f})")
print("\nAll plots saved. Project complete!")
