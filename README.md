# agentic-ds-framework

Six-agent LangGraph data-science pipeline (profiling, cleaning, feature engineering, model selection, evaluation, reflection), compared against a manual scikit-learn baseline and H2O AutoML. Results are logged to MLflow (`agentic-ds-framework`).

## AutoDS app

`autods_dashboard.py` is a Streamlit front end for the agentic pipeline: **Setup** (upload a CSV, pick a binary target, set loop limits), **Live Run** (per-agent status, AUC-ROC and F1 per iteration, the reflection agent's diagnosis) and **Reports** (summary metrics, iteration chart, PDF export, model config).

```
pip install -r requirements.txt
streamlit run autods_dashboard.py
```

- Runs use your own Anthropic API key, entered in the sidebar. It is passed only to your run's worker process and never written to disk.
- "Watch a recorded run" replays a real logged Telco churn run (seed 42) without an API key.
- Uploaded data is read once by the worker and deleted from disk; at most two live runs execute at a time.
- "Download Model Config" exports the selected pipeline's specification (cleaning and feature plans, model, hyperparameters, threshold). A fitted model object is not persisted.

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. At <https://share.streamlit.io>, choose **Create app**, select the repository and branch `main`, and set the main file to `autods_dashboard.py`.
3. Under **Advanced settings**, select Python 3.12.

## Experiments

`run_experiments.py`, `baseline_manual.py` and `baseline_h2o.py` reproduce the thesis grid (Telco churn, credit default, stroke; seeds 42, 123, 2024). Tests: `python test_*.py`.
