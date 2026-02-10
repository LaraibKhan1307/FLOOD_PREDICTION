import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import torch.nn as nn
import torch
# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Flood Prediction Dashboard", layout="wide")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")


class ProbabilisticMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(32, 1)
        self.log_var_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.net(x)
        mean = torch.sigmoid(self.mean_head(h))
        log_var = self.log_var_head(h)
        return mean, log_var



df = load_data()

target_col = df.columns[-1]
X = df.drop(columns=[target_col]).iloc[:, 1:]   # drop ID/index column
y = df[target_col]

# =====================================================
# LOAD PRE-TRAINED MODELS (NO TRAINING)
# =====================================================
@st.cache_resource
def load_models():
    models = {
        "Linear Regression": load("models/linear_model.joblib"),
        "Ridge": load("models/ridge_model.joblib"),
        "Lasso": load("models/lasso_model.joblib"),
        "ElasticNet": load("models/elasticnet_model.joblib"),
        "Decision Tree": load("models/dt_model.joblib"),
    }
    scaler = load("models/scaler.joblib")

    mlp = ProbabilisticMLP(input_dim=X.shape[1])
    mlp.load_state_dict(torch.load("models/mlp_model.pt", map_location="cpu"))
    mlp.eval()

    models["MLP"] = mlp
    return models, scaler

models, scaler = load_models()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", ["EDA", "Flood Prediction"])

# =====================================================
# EDA SECTION
# =====================================================
if page == "EDA":

    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.histplot(y, bins=30, kde=True, ax=ax)
        ax.set_xlabel("Flood Probability")
        st.pyplot(fig)

    with col2:
        st.subheader("Flood vs No-Flood (Threshold = 0.5)")
        y_bin = (y >= 0.5).astype(int)
        fig, ax = plt.subplots()
        sns.countplot(x=y_bin, ax=ax)
        ax.set_xticklabels(["No Flood", "Flood"])
        st.pyplot(fig)

    st.subheader("Feature Correlation with Target")
    corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(corr.to_frame(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions")

    selected_features = st.multiselect(
        "Select features to visualize",
        X.columns.tolist(),
        default=X.columns[:3].tolist()
    )

    if selected_features:
        fig, axes = plt.subplots(
            1, len(selected_features),
            figsize=(5 * len(selected_features), 4)
        )

        if len(selected_features) == 1:
            axes = [axes]

        for ax, feat in zip(axes, selected_features):
            sns.histplot(X[feat], bins=30, ax=ax)
            ax.set_title(feat)

        st.pyplot(fig)

# =====================================================
# PREDICTION SECTION
# =====================================================
if page == "Flood Prediction":

    st.title("🌊 Flood Prediction (Pre-trained Models)")

    model_name = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    st.subheader("Enter Input Features")

    user_input = []
    cols = st.columns(3)

    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            val = st.number_input(
                col,
                value=float(X[col].mean())
            )
            user_input.append(val)

    if st.button("Predict Flood Probability"):

        user_input = np.array(user_input).reshape(1, -1)


        if model_name in ["Decision Tree"]:
            prediction = models[model_name].predict(user_input)[0]

        elif model_name == "MLP":
            user_input_scaled = scaler.transform(user_input)
            x_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

            with torch.no_grad():
                mean, log_var = models["MLP"](x_tensor)

            prediction = mean.item()
            uncertainty = torch.exp(0.5 * log_var).item()

            st.info(f"📉 Prediction Uncertainty (σ): {uncertainty:.3f}")

        else:
            user_input_scaled = scaler.transform(user_input)
            prediction = models[model_name].predict(user_input_scaled)[0]


        st.success(f"🌊 Predicted Flood Probability: **{prediction:.3f}**")

        if prediction >= 0.5:
            st.error("⚠️ High Flood Risk")
        else:
            st.success("✅ Low Flood Risk")
