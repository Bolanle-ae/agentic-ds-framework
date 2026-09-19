import os; os.environ["ANTHROPIC_API_KEY"] = input("Enter your Anthropic API key: ")

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

def profile_csv(filepath: str) -> dict:
    df = pd.read_csv(filepath)
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().mean() * 100).round(2).to_dict(),
        "numeric_summary": df.describe().round(3).to_dict(),
        "class_balance": None
    }
    for col in ["Churn", "default", "label", "target", "y"]:
        if col in df.columns:
            profile["class_balance"] = df[col].value_counts(normalize=True).round(3).to_dict()
            break
    return profile

filepath = "/Users/admin/Desktop/AutoML/Customer-Churn.csv"
profile = profile_csv(filepath)

messages = [
    SystemMessage(content="You are a data analysis assistant. Provide a clear, concise summary in plain English."),
    HumanMessage(content=f"Please summarise this dataset profile:\n{profile}")
]

result = llm.invoke(messages)
print(result.content)
