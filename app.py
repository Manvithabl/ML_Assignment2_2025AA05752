import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="centered"
)

st.title("Breast Cancer Classification App")
st.write("Upload a test CSV file, select a model, and view evaluation results.")

# ---------------------------------
# Load saved preprocessing objects
# ---------------------------------
imputer = joblib.load("model/imputer.pkl")
scaler = joblib.load("model/scaler.pkl")

# Available models
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# ---------------------------------
# Dataset upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload TEST dataset (CSV format)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Drop unused columns if present
    df.drop(columns=["id", "Unnamed: 32"], errors="ignore", inplace=True)

    # Check target column
    if "diagnosis" not in df.columns:
        st.error("CSV must contain a 'diagnosis' column.")
        st.stop()

    # Separate features and target
    X = df.drop("diagnosis", axis=1)
    y_true = df["diagnosis"].map({"M": 1, "B": 0})

    # ---------------------------------
    # Model selection
    # ---------------------------------
    model_name = st.selectbox(
        "Select Classification Model",
        list(MODEL_FILES.keys())
    )

    model = joblib.load(f"model/{MODEL_FILES[model_name]}")

    # ---------------------------------
    # Preprocessing
    # ---------------------------------
    X_imputed = imputer.transform(X)

    if model_name in ["Decision Tree", "Random Forest"]:
        X_final = X_imputed
    else:
        X_final = scaler.transform(X_imputed)

    # ---------------------------------
    # Prediction
    # ---------------------------------
    y_pred = model.predict(X_final)

    # ---------------------------------
    # Metrics
    # ---------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", round(accuracy, 4))
        st.metric("Precision", round(precision, 4))

    with col2:
        st.metric("Recall", round(recall, 4))
        st.metric("F1 Score", round(f1, 4))

    # ---------------------------------
    # Confusion Matrix
    # ---------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")
