import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage


st.set_page_config(page_title="AutoML Pipeline", layout="wide")
st.title("AutoML Pipeline")
st.caption("Automated profiling, cleaning, feature engineering, modelling, and reflection.")


with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    run_btn = st.button(
        "Run Pipeline",
        type="primary",
        disabled=not (api_key and uploaded_file)
    )


def run_profiler(filepath: str) -> dict:
    df = pd.read_csv(filepath)
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().mean() * 100).round(2).to_dict(),
        "target_col": None,
        "class_balance": None,
    }
    for col in ["Churn", "default", "label", "target", "y"]:
        if col in df.columns:
            profile["target_col"] = col
            profile["class_balance"] = df[col].value_counts(normalize=True).round(3).to_dict()
            break
    return profile


def run_cleaning(filepath: str):
    df = pd.read_csv(filepath)
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df), df.isnull().sum().sum()


def run_feature_engineering(df: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    if target_col and df[target_col].dtype == "object":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
    for col in df.select_dtypes(include=["object"]).columns.tolist():
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    return df


def run_model(df: pd.DataFrame, target_col: str) -> float:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return round(accuracy_score(y_test, model.predict(X_test)), 4)


def run_reflection(profile: dict, accuracy: Optional[float], llm: ChatAnthropic) -> str:
    prompt = f"""You are an expert data scientist reviewing an automated ML pipeline run.

Pipeline results:
- Dataset: {profile['row_count']} rows, {profile['column_count']} columns
- Null percentages: {profile['null_percentages']}
- Class balance: {profile.get('class_balance', 'unknown')}
- Model: Logistic Regression
- Accuracy: {accuracy}

Identify the single most likely reason the accuracy is not higher, and suggest
one concrete improvement to the pipeline. Be specific and brief."""
    return llm.invoke([HumanMessage(content=prompt)]).content


if run_btn:
    os.environ["ANTHROPIC_API_KEY"] = api_key
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.status("Running pipeline...", expanded=True) as status:
            st.write("Node 1: Profiling dataset...")
            profile = run_profiler(tmp_path)

            st.write("Node 2: Cleaning dataset...")
            df_cleaned, dupes_removed, remaining_nulls = run_cleaning(tmp_path)

            st.write("Node 3: Engineering features...")
            target_col = profile.get("target_col")
            df_features = run_feature_engineering(df_cleaned, target_col)

            st.write("Node 4: Training model...")
            accuracy = run_model(df_features, target_col) if target_col else None

            st.write("Node 5: Reflecting with Claude...")
            reflection = run_reflection(profile, accuracy, llm)

            status.update(label="Pipeline complete!", state="complete")

        st.header("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{profile['row_count']:,}")
        col2.metric("Columns", profile["column_count"])
        col3.metric("Accuracy", f"{accuracy * 100:.1f}%" if accuracy else "N/A")

        with st.expander("Node 1 — Dataset Profile", expanded=True):
            null_df = pd.DataFrame({
                "Column": list(profile["null_counts"].keys()),
                "Dtype": list(profile["dtypes"].values()),
                "Null Count": list(profile["null_counts"].values()),
                "Null %": list(profile["null_percentages"].values()),
            })
            st.dataframe(null_df, use_container_width=True)
            if profile.get("class_balance"):
                st.subheader(f"Class Balance — {profile['target_col']}")
                st.bar_chart(profile["class_balance"])

        with st.expander("Node 2 — Cleaning Summary"):
            c1, c2 = st.columns(2)
            c1.metric("Duplicates Removed", dupes_removed)
            c2.metric("Remaining Nulls", int(remaining_nulls))

        with st.expander("Node 3 — Feature Engineering"):
            st.write(f"**{len(df_features.columns) - 1}** input features prepared for training.")

        with st.expander("Node 4 — Model"):
            if accuracy:
                st.metric("Logistic Regression Accuracy", f"{accuracy * 100:.1f}%")
            else:
                st.warning("No target column detected — model training was skipped.")

        with st.expander("Node 5 — Claude's Reflection", expanded=True):
            st.info(reflection)

    finally:
        os.unlink(tmp_path)
