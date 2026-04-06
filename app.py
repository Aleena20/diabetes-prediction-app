# -*- coding: utf-8 -*-
"""
Diabetes Prediction - Streamlit App
=====================================
Run with: streamlit run app.py
"""

import streamlit as st
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
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="assets/favicon.png" if True else ":pill:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSlider label {
    font-size: 0.8rem;
    color: #94a3b8 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 8px;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #0f172a;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 6px;
}

/* Prediction result */
.result-diabetic {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 1.5px solid #fca5a5;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-healthy {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    margin-bottom: 6px;
}
.result-sub {
    font-size: 0.9rem;
    color: #475569;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 0.875rem;
    padding: 8px 20px;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    color: #0f172a !important;
    background: #f8fafc;
    border-radius: 6px 6px 0 0;
}

/* Section header */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #0f172a;
    margin: 0 0 4px 0;
}
.section-sub {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan).fillna(df[col].median())
    return df


@st.cache_resource
def train_models(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy").mean()
        results[name] = {
            "model":      model,
            "accuracy":   accuracy_score(y_test, y_pred),
            "roc_auc":    roc_auc_score(y_test, y_prob),
            "cv":         cv,
            "y_pred":     y_pred,
            "y_prob":     y_prob,
            "y_test":     y_test,
        }
    return scaler, X.columns.tolist(), results, X_test, y_test


df = load_data()
scaler, feature_names, model_results, X_test, y_test = train_models(df)


# ─────────────────────────────────────────────
# SIDEBAR — Patient Input
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Patient Data")
    st.markdown("Adjust the sliders to enter patient measurements.")
    st.markdown("---")

    pregnancies = st.slider("Pregnancies",        0,  17,  3)
    glucose     = st.slider("Glucose (mg/dL)",   44, 199, 120)
    bp          = st.slider("Blood Pressure",     0, 122,  70)
    skin        = st.slider("Skin Thickness (mm)",0,  99,  20)
    insulin     = st.slider("Insulin (mu U/ml)",  0, 846,  80)
    bmi         = st.slider("BMI",              10.0, 67.1, 31.5, step=0.1)
    dpf         = st.slider("Diabetes Pedigree", 0.078, 2.42, 0.47, step=0.001)
    age         = st.slider("Age",               21,  81,  33)

    st.markdown("---")
    model_choice = st.selectbox(
        "Model",
        ["Random Forest", "Logistic Regression", "K-Nearest Neighbours"]
    )
    st.markdown("---")
    predict_btn = st.button("Run Prediction", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown('<p class="section-header" style="font-size:2.4rem;">Diabetes Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Pima Indians Diabetes Dataset &middot; 768 patients &middot; 8 clinical features</p>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediction", "Exploratory Analysis", "Model Comparison"])


# ══════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════
with tab1:
    if predict_btn:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        chosen = model_results[model_choice]
        model  = chosen["model"]
        pred   = model.predict(input_scaled)[0]
        prob   = model.predict_proba(input_scaled)[0]

        diabetic_prob  = prob[1] * 100
        healthy_prob   = prob[0] * 100

        st.markdown("### Prediction Result")
        col_res, col_gauge = st.columns([1, 1], gap="large")

        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="result-diabetic">
                    <div class="result-title" style="color:#dc2626;">Diabetic Risk Detected</div>
                    <div class="result-sub">The model predicts a <strong>{diabetic_prob:.1f}%</strong>
                    probability of diabetes based on the input values.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-healthy">
                    <div class="result-title" style="color:#16a34a;">Low Risk — Non-Diabetic</div>
                    <div class="result-sub">The model predicts a <strong>{healthy_prob:.1f}%</strong>
                    probability of being non-diabetic based on the input values.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Model used:** {model_choice}")

        with col_gauge:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            categories = ["Non-Diabetic", "Diabetic"]
            values     = [healthy_prob, diabetic_prob]
            colors     = ["#4ade80", "#f87171"]
            bars = ax.barh(categories, values, color=colors, height=0.5, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=12, fontweight="bold", color="#0f172a")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)", fontsize=10, color="#64748b")
            ax.set_title("Prediction Probabilities", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
            ax.tick_params(colors="#475569")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_facecolor("#f8fafc")
            fig.patch.set_facecolor("#f8fafc")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Input summary table
        st.markdown("---")
        st.markdown("#### Input Summary")
        input_df = pd.DataFrame({
            "Feature": feature_names,
            "Your Value": [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age],
            "Dataset Median": df[feature_names].median().round(2).values,
        })
        input_df["vs. Median"] = (input_df["Your Value"] - input_df["Dataset Median"]).round(2)
        st.dataframe(input_df.set_index("Feature"), use_container_width=True)

    else:
        st.info("Adjust the patient sliders in the sidebar and click **Run Prediction** to see results.")

        # Show top-level dataset stats while waiting
        st.markdown("#### Dataset at a Glance")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-card"><div class="label">Patients</div><div class="value">768</div><div class="sub">total records</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div class="label">Features</div><div class="value">8</div><div class="sub">clinical inputs</div></div>', unsafe_allow_html=True)
        with c3:
            pct = round(df["Outcome"].mean() * 100, 1)
            st.markdown(f'<div class="metric-card"><div class="label">Diabetic</div><div class="value">{pct}%</div><div class="sub">of patients</div></div>', unsafe_allow_html=True)
        with c4:
            best = max(model_results, key=lambda k: model_results[k]["roc_auc"])
            best_auc = round(model_results[best]["roc_auc"], 3)
            st.markdown(f'<div class="metric-card"><div class="label">Best AUC</div><div class="value">{best_auc}</div><div class="sub">{best.split()[0]} Forest</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Visualising how each feature distributes across diabetic and non-diabetic patients.</p>', unsafe_allow_html=True)

    # Feature distributions
    selected_feature = st.selectbox("Select a feature to inspect", feature_names)

    col_dist, col_box = st.columns(2)
    with col_dist:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[df["Outcome"] == 0][selected_feature].hist(
            ax=ax, bins=25, alpha=0.7, color="#60a5fa", label="Non-Diabetic", edgecolor="white")
        df[df["Outcome"] == 1][selected_feature].hist(
            ax=ax, bins=25, alpha=0.7, color="#f87171", label="Diabetic", edgecolor="white")
        ax.set_title(f"{selected_feature} — Distribution", fontweight="bold", color="#0f172a")
        ax.set_xlabel(selected_feature, color="#475569")
        ax.set_ylabel("Count", color="#475569")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_box:
        fig, ax = plt.subplots(figsize=(6, 4))
        data_0 = df[df["Outcome"] == 0][selected_feature]
        data_1 = df[df["Outcome"] == 1][selected_feature]
        bp_plot = ax.boxplot(
            [data_0, data_1],
            labels=["Non-Diabetic", "Diabetic"],
            patch_artist=True,
            medianprops=dict(color="#0f172a", linewidth=2),
        )
        bp_plot["boxes"][0].set_facecolor("#bfdbfe")
        bp_plot["boxes"][1].set_facecolor("#fecaca")
        ax.set_title(f"{selected_feature} — Box Plot", fontweight="bold", color="#0f172a")
        ax.set_ylabel(selected_feature, color="#475569")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Correlation heatmap
    st.markdown("---")
    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, mask=mask, ax=ax,
                linewidths=0.5, linecolor="#e2e8f0",
                annot_kws={"size": 9})
    ax.set_title("Feature Correlations (lower triangle)", fontweight="bold", color="#0f172a", pad=12)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Raw data toggle
    st.markdown("---")
    if st.checkbox("Show raw dataset"):
        st.dataframe(df, use_container_width=True, height=300)


# ══════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Logistic Regression vs Random Forest vs K-Nearest Neighbours</p>', unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for col, (name, res) in zip(cols, model_results.items()):
        short = name.replace("K-Nearest Neighbours", "KNN")
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{short}</div>
                <div class="value">{res['roc_auc']:.3f}</div>
                <div class="sub">ROC-AUC &nbsp;|&nbsp; Acc: {res['accuracy']:.3f} &nbsp;|&nbsp; CV: {res['cv']:.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_roc, col_fi = st.columns(2)

    # ROC Curves
    with col_roc:
        fig, ax = plt.subplots(figsize=(6, 5))
        palette = ["#3b82f6", "#ef4444", "#22c55e"]
        for (name, res), color in zip(model_results.items(), palette):
            fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
            ax.plot(fpr, tpr, label=f"{name} ({res['roc_auc']:.3f})", color=color, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate", color="#475569")
        ax.set_ylabel("True Positive Rate", color="#475569")
        ax.set_title("ROC Curves", fontweight="bold", color="#0f172a")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Feature Importance (RF)
    with col_fi:
        rf_model = model_results["Random Forest"]["model"]
        importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        colors_bar = ["#bfdbfe" if i < len(importances) - 3 else "#3b82f6" for i in range(len(importances))]
        importances.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="white")
        ax.set_title("Feature Importance (Random Forest)", fontweight="bold", color="#0f172a")
        ax.set_xlabel("Importance Score", color="#475569")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("#f8fafc")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Confusion Matrices
    st.markdown("---")
    st.markdown("#### Confusion Matrices")
    cols_cm = st.columns(3)
    for col, (name, res) in zip(cols_cm, model_results.items()):
        with col:
            cm = confusion_matrix(res["y_test"], res["y_pred"])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Non-Diab.", "Diabetic"],
                        yticklabels=["Non-Diab.", "Diabetic"],
                        linewidths=0.5, linecolor="white")
            ax.set_title(name.replace("K-Nearest Neighbours", "KNN"),
                         fontsize=10, fontweight="bold", color="#0f172a")
            ax.set_xlabel("Predicted", color="#475569", fontsize=9)
            ax.set_ylabel("Actual", color="#475569", fontsize=9)
            fig.patch.set_facecolor("#f8fafc")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # Full classification reports
    st.markdown("---")
    with st.expander("Full Classification Reports"):
        for name, res in model_results.items():
            st.markdown(f"**{name}**")
            report = classification_report(
                res["y_test"], res["y_pred"],
                target_names=["Non-Diabetic", "Diabetic"]
            )
            st.code(report)
