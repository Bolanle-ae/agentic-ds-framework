import os
os.environ["ANTHROPIC_API_KEY"] = input("Enter your Anthropic API key: ")

import pandas as pd
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


class PipelineState(TypedDict):
    filepath: str
    profile: dict
    cleaning_plan: str


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
        "numeric_summary": df.describe().round(3).to_dict(),
    }

    for col in ["Churn", "default", "label", "target", "y"]:
        if col in df.columns:
            profile["class_balance"] = df[col].value_counts(normalize=True).round(3).to_dict()
            break

    print(f"  Rows: {profile['row_count']} | Columns: {profile['column_count']}")
    print(f"  Nulls detected: { {k: v for k, v in profile['null_counts'].items() if v > 0} }")

    return {"profile": profile}


def cleaning_planner_node(state: PipelineState) -> dict:
    print("\n[Node 2] Generating cleaning plan with Claude...")

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    profile = state["profile"]

    prompt = f"""You are a data cleaning expert. Based on the dataset profile below, 
produce a concise cleaning plan. List specific actions only — no fluff.

Dataset Profile:
- Rows: {profile['row_count']}
- Columns: {profile['column_count']}
- Column names: {profile['columns']}
- Data types: {profile['dtypes']}
- Null counts: {profile['null_counts']}
- Null percentages: {profile['null_percentages']}
- Numeric summary: {profile['numeric_summary']}
{f"- Class balance: {profile.get('class_balance')}" if profile.get('class_balance') else ""}

Cleaning plan:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    cleaning_plan = response.content

    print("\n  Cleaning Plan:")
    print(cleaning_plan)

    return {"cleaning_plan": cleaning_plan}


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("profiler", profiler_node)
    graph.add_node("cleaning_planner", cleaning_planner_node)

    graph.set_entry_point("profiler")
    graph.add_edge("profiler", "cleaning_planner")
    graph.add_edge("cleaning_planner", END)

    return graph.compile()


if __name__ == "__main__":
    pipeline = build_pipeline()

    initial_state = {
        "filepath": "/Users/admin/Desktop/AutoML/Customer-Churn.csv",
        "profile": {},
        "cleaning_plan": ""
    }

    final_state = pipeline.invoke(initial_state)

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print("\nFinal cleaning plan saved to state.")
