import os
os.environ["ANTHROPIC_API_KEY"] = input("Enter your Anthropic API key: ")

import pandas as pd
import numpy as np
from typing import TypedDict, Optional

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


class PipelineState(TypedDict):
    filepath: str
    profile: dict
    df_cleaned: Optional[object]
    df_features: Optional[object]
    accuracy: Optional[float]
    cleaning_plan: str
    reflection: str


def profiler_node(state: PipelineState) -> dict:
    print("\n[Node 1] Profiling dataset...")
    df = pd.read_csv(state["filepath"])

    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().mean() * 100).round(2).to_dict(),
    }

    for col in ["Churn", "default", "label", "target", "y"]:
        if col in df.columns:
            profile["target_col"] = col
            profile["class_balance"] = df[col].value_counts(normalize=True).round(3).to_dict()
            break

    print(f"  Rows: {profile['row_count']} | Columns: {profile['column_count']}")
    return {"profile": profile}


def cleaning_node(state: PipelineState) -> dict:
    print("\n[Node 2] Cleaning dataset...")
    df = pd.read_csv(state["filepath"])

    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {before - len(df)}")
    print(f"  Remaining nulls: {df.isnull().sum().sum()}")

    return {"df_cleaned": df}


def feature_engineering_node(state: PipelineState) -> dict:
    print("\n[Node 3] Engineering features...")
    df = state["df_cleaned"].copy()

    target_col = state["profile"].get("target_col", None)

    if target_col and df[target_col].dtype == "object":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"  Features ready: {len(df.columns) - 1} input features")
    return {"df_features": df}


def model_node(state: PipelineState) -> dict:
    print("\n[Node 4] Training model...")
    df = state["df_features"].copy()
    target_col = state["profile"].get("target_col")

    if not target_col:
        print("  No target column found — skipping model training.")
        return {"accuracy": None}

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, preds), 4)

    print(f"  Accuracy: {accuracy}")
    return {"accuracy": accuracy}


def reflection_node(state: PipelineState) -> dict:
    print("\n[Node 5] Reflecting on results...")

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    prompt = f"""You are an expert data scientist reviewing an automated ML pipeline run.

Pipeline results:
- Dataset: {state['profile']['row_count']} rows, {state['profile']['column_count']} columns
- Null percentages: {state['profile']['null_percentages']}
- Class balance: {state['profile'].get('class_balance', 'unknown')}
- Model: Logistic Regression
- Accuracy: {state['accuracy']}

Identify the single most likely reason the accuracy is not higher, and suggest 
one concrete improvement to the pipeline. Be specific and brief."""

    response = llm.invoke([HumanMessage(content=prompt)])
    reflection = response.content

    print(f"\n  Reflection:\n{reflection}")
    return {"reflection": reflection}


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("profiler", profiler_node)
    graph.add_node("cleaner", cleaning_node)
    graph.add_node("feature_engineer", feature_engineering_node)
    graph.add_node("model", model_node)
    graph.add_node("reflection", reflection_node)

    graph.set_entry_point("profiler")
    graph.add_edge("profiler", "cleaner")
    graph.add_edge("cleaner", "feature_engineer")
    graph.add_edge("feature_engineer", "model")
    graph.add_edge("model", "reflection")
    graph.add_edge("reflection", END)

    return graph.compile()


if __name__ == "__main__":
    pipeline = build_pipeline()

    initial_state = {
        "filepath": "/Users/admin/Desktop/AutoML/Customer-Churn.csv",
        "profile": {},
        "df_cleaned": None,
        "df_features": None,
        "accuracy": None,
        "cleaning_plan": "",
        "reflection": ""
    }

    final_state = pipeline.invoke(initial_state)

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"\nAccuracy: {final_state['accuracy']}")
    print(f"\nReflection:\n{final_state['reflection']}")
