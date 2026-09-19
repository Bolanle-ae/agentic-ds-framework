"""Six-agent LangGraph pipeline (profiler, cleaner, feature engineer, model selector, evaluator, reflection) with Optuna tuning inside the model selector."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import numpy as np
import optuna
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from optuna.samplers import TPESampler
from pydantic import BaseModel, ValidationError
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from experiment_common import (
    DATASETS,
    RunResult,
    SIX_STAGES,
    THRESHOLD_CALIBRATION_METRIC,
    calibrate_threshold_oof,
    compute_metrics,
    encode_target,
    ensure_api_key,
    load_dataset,
    log_run,
    make_split,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

LLM_MODEL = "claude-haiku-4-5"
# Defaults to LLM_MODEL (Haiku); set the env var to swap the model for the reflection agent only.
REFLECTION_LLM_MODEL = os.environ.get("REFLECTION_LLM_MODEL", LLM_MODEL)
HAIKU_PRICE_PER_MTOK = {"input": 1.00, "output": 5.00}

CANDIDATE_MODELS = [
    "logistic_regression", "random_forest", "gradient_boosting",
    "extra_trees", "svm", "knn", "naive_bayes", "decision_tree", "adaboost",
    "xgboost", "lightgbm", "catboost",
]

MAX_ITERATIONS_FULL = 5
# Two iterations under --dry-run so at least one retry fires; full runs use MAX_ITERATIONS_FULL.
MAX_ITERATIONS_DRY_RUN = 2
PERFORMANCE_TARGET_METRIC = "auc_roc"
PERFORMANCE_TARGET = 0.85
NO_IMPROVEMENT_EPSILON = 0.001
NO_IMPROVEMENT_PATIENCE = 3

# AUC-ROC gaps below this are within test-set sampling noise, so near-ties are broken by positive-class F1.
BEST_ITERATION_AUC_TOLERANCE = 0.01

OVERFITTING_GAP_THRESHOLD = 0.10
UNDERFITTING_AUC_THRESHOLD = 0.65

# A re-tune after an upstream change starts from a known-good family, so it needs fewer trials than a first tune.
OPTUNA_TRIALS_FIRST = 25
OPTUNA_TRIALS_RETUNE = 10

# Optuna tunes on a stratified subsample above this row count; the final model is always fit on the full training set.
OPTUNA_SUBSAMPLE_ROW_THRESHOLD = 10_000
OPTUNA_SUBSAMPLE_SIZE = 5_000

TARGET_ENCODING_FOLDS = 5
TARGET_ENCODING_SMOOTHING = 10.0

FEATURE_SELECTION_VARIANCE_THRESHOLD = 1e-4

MISSINGNESS_MIN_MISSING_ROWS = 5
MISSINGNESS_CONCENTRATION_THRESHOLD = 0.90
MISSINGNESS_BASE_RATE_MARGIN = 0.15
ARITHMETIC_MIN_SUPPORT_ROWS = 30
# Real relationships (e.g. Telco TotalCharges ~ tenure x MonthlyCharges) deviate 5-10% on some rows, so a stricter 2%/95% missed them.
ARITHMETIC_MATCH_TOLERANCE = 0.10
ARITHMETIC_MIN_MATCH_RATE = 0.90


# class_weight_ratio (negative:positive) is used by xgboost/lightgbm; sklearn models get 'balanced'; models without the option ignore it.
def build_model(name: str, params: dict, seed: int, class_weight_ratio: Optional[float] = None):
    weighted = class_weight_ratio is not None
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced" if weighted else None, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, class_weight="balanced" if weighted else None, **params)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **params)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, class_weight="balanced" if weighted else None, **params)
    if name == "svm":
        return SVC(probability=True, random_state=seed, class_weight="balanced" if weighted else None, **params)
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "naive_bayes":
        return GaussianNB(**params)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, class_weight="balanced" if weighted else None, **params)
    if name == "adaboost":
        return AdaBoostClassifier(random_state=seed, **params)
    if name == "xgboost":
        extra = {"scale_pos_weight": class_weight_ratio} if weighted else {}
        return XGBClassifier(random_state=seed, eval_metric="logloss", **extra, **params)
    if name == "lightgbm":
        extra = {"scale_pos_weight": class_weight_ratio} if weighted else {}
        return LGBMClassifier(random_state=seed, verbosity=-1, **extra, **params)
    if name == "catboost":
        extra = {"auto_class_weights": "Balanced"} if weighted else {}
        return CatBoostClassifier(random_state=seed, verbose=False, **extra, **params)
    raise ValueError(f"Unknown model name: {name}")


def suggest_params(trial: "optuna.Trial", name: str) -> dict:
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
    if name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
        }
    if name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
    if name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    if name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    if name == "naive_bayes":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)}
    if name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    if name == "adaboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    raise ValueError(f"Unknown model name: {name}")


def _run_optuna(X, y, model_name: str, seed: int, n_trials: int, class_weight_ratio: Optional[float] = None):
    def objective(trial):
        params = suggest_params(trial, model_name)
        model = build_model(model_name, params, seed, class_weight_ratio)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=1)
        return scores.mean()

    try:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params, True
    except Exception:
        return {}, False


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def call_llm_json(llm: ChatAnthropic, system_prompt: str, user_prompt: str):
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
    except Exception as e:
        return None, {"input_tokens": 0, "output_tokens": 0}, f"LLM call failed: {type(e).__name__}: {e}"

    usage = response.usage_metadata or {}
    usage_dict = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    parsed = _extract_json(response.content)
    if parsed is None:
        return None, usage_dict, f"Could not parse JSON from LLM response: {response.content[:200]!r}"
    return parsed, usage_dict, None


def _sum_usage(a: dict, b: dict) -> dict:
    return {
        "input_tokens": a.get("input_tokens", 0) + b.get("input_tokens", 0),
        "output_tokens": a.get("output_tokens", 0) + b.get("output_tokens", 0),
    }


def _add_agent_tokens(token_usage: dict, agent: str, usage: dict) -> dict:
    token_usage = {k: dict(v) for k, v in token_usage.items()}
    bucket = dict(token_usage.get(agent, {"input_tokens": 0, "output_tokens": 0}))
    bucket["input_tokens"] = bucket.get("input_tokens", 0) + usage.get("input_tokens", 0)
    bucket["output_tokens"] = bucket.get("output_tokens", 0) + usage.get("output_tokens", 0)
    token_usage[agent] = bucket
    return token_usage


def _token_cost_usd(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * HAIKU_PRICE_PER_MTOK["input"]
        + usage.get("output_tokens", 0) / 1e6 * HAIKU_PRICE_PER_MTOK["output"]
    )


class AgenticState(TypedDict):
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_col: str
    positive_label: Any
    dataset_name: str
    seed: int
    max_iterations: int
    optuna_trials: Optional[int]
    dry_run: bool
    performance_target: Optional[float]
    no_improvement_epsilon: float

    profile: Dict[str, Any]

    cleaning_plan: Optional[dict]
    feature_plan: Optional[dict]

    X_train_clean: Optional[pd.DataFrame]
    X_test_clean: Optional[pd.DataFrame]
    y_train: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    X_train_features: Optional[pd.DataFrame]
    X_test_features: Optional[pd.DataFrame]

    model_name: Optional[str]
    model: Optional[Any]
    best_params: Optional[dict]
    train_time_sec: Optional[float]
    tuning_case: Optional[str]
    use_class_weighting: bool
    class_weight_ratio: Optional[float]
    metrics: Optional[dict]
    threshold: Optional[float]
    calibration_metric: str

    iteration: int
    iteration_history: List[Dict[str, Any]]
    stage_automation: Dict[str, bool]
    events: List[Dict[str, str]]
    used_fallback: bool
    token_usage: Dict[str, Dict[str, int]]
    stage_timings: List[Dict[str, Any]]
    optuna_durations: List[float]

    weakest_component: Optional[str]
    current_instruction: Optional[str]
    current_failure_mode: Optional[str]
    stop_reason: Optional[str]
    no_improvement_streak: int

    tried_models: List[str]


# True on the first pass or when reflection targeted this stage; otherwise the stored plan is replayed unchanged.
def _is_targeted(state: AgenticState, stage_name: str) -> bool:
    return state["weakest_component"] is None or state["weakest_component"] == stage_name


SENTINEL_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "unknown", "?", "-", "--", "missing"}


def _count_disguised_nulls(series: pd.Series) -> int:
    stripped = series.dropna().astype(str).str.strip().str.lower()
    return int(stripped.isin(SENTINEL_MISSING_TOKENS).sum())


def _detect_date_like_columns(features: pd.DataFrame) -> dict:
    result = {}
    non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        series = features[col].dropna()
        if series.empty:
            continue
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        parse_rate = parsed.notna().mean()
        if parse_rate < 0.9:
            continue
        valid = parsed.dropna()
        result[col] = {
            "parse_rate": round(float(parse_rate), 3),
            "date_range": [str(valid.min().date()), str(valid.max().date())],
            "nunique_full_date": int(valid.dt.date.nunique()),
            "nunique_month": int(valid.dt.to_period("M").nunique()),
            "nunique_year": int(valid.dt.year.nunique()),
            "nunique_day_of_week": int(valid.dt.dayofweek.nunique()),
        }
    return result


# Discovery-only view: blank/sentinel tokens count as missing and numeric-parsable text as numeric; the data is not modified.
def _coerce_effectively_numeric(features: pd.DataFrame, min_parse_rate: float = 0.9) -> pd.DataFrame:
    coerced = features.copy()
    for col in features.select_dtypes(exclude=[np.number]).columns:
        as_str = coerced[col].astype(str).str.strip()
        disguised = as_str.str.lower().isin(SENTINEL_MISSING_TOKENS)
        candidate = coerced[col].where(~disguised, np.nan)
        numeric_candidate = pd.to_numeric(candidate, errors="coerce")
        non_missing = candidate.notna()
        if non_missing.sum() == 0:
            continue
        parse_rate = numeric_candidate[non_missing].notna().mean()
        if parse_rate >= min_parse_rate:
            coerced[col] = numeric_candidate
    return coerced


# Every category of a driving column is checked, not just the mode, so rare values like 'Unknown' can be found.
MISSINGNESS_MAX_CATEGORIES_CHECKED = 20


def _discover_missingness_correlations(features: pd.DataFrame) -> dict:
    findings: Dict[str, list] = {}
    missing_cols = [c for c in features.columns if features[c].isnull().sum() >= MISSINGNESS_MIN_MISSING_ROWS]
    for target in missing_cols:
        missing_mask = features[target].isnull()
        candidates = []
        for other in features.columns:
            if other == target:
                continue
            col = features[other]
            if pd.api.types.is_numeric_dtype(col):
                check_values = [0]
            else:
                uniques = col.dropna().unique().tolist()
                if len(uniques) > MISSINGNESS_MAX_CATEGORIES_CHECKED:
                    continue
                check_values = uniques
            for check_value in check_values:
                base_rate = float((col == check_value).mean())
                if base_rate <= 0 or base_rate >= 0.98:
                    continue
                concentration = float((col[missing_mask] == check_value).mean())
                if concentration >= MISSINGNESS_CONCENTRATION_THRESHOLD and concentration - base_rate >= MISSINGNESS_BASE_RATE_MARGIN:
                    candidates.append({
                        "column": other,
                        "value": float(check_value) if pd.api.types.is_numeric_dtype(col) else str(check_value),
                        "concentration_in_missing_rows": round(concentration, 3),
                        "base_rate_overall": round(base_rate, 3),
                    })
        if candidates:
            findings[target] = sorted(candidates, key=lambda c: -(c["concentration_in_missing_rows"] - c["base_rate_overall"]))
    return findings


def _discover_arithmetic_relationships(features: pd.DataFrame) -> dict:
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    findings: Dict[str, dict] = {}
    for target in numeric_cols:
        missing = features[target].isnull()
        if missing.sum() == 0:
            continue
        observed = features.loc[~missing]
        if len(observed) < ARITHMETIC_MIN_SUPPORT_ROWS:
            continue
        target_vals = observed[target].to_numpy(dtype=float)
        denom = np.clip(np.abs(target_vals), 1e-6, None)
        other_numeric = [c for c in numeric_cols if c != target]
        best = None
        for i in range(len(other_numeric)):
            for j in range(i + 1, len(other_numeric)):
                a, b = other_numeric[i], other_numeric[j]
                a_vals = observed[a].to_numpy(dtype=float)
                b_vals = observed[b].to_numpy(dtype=float)
                for method, combined in (("sum", a_vals + b_vals), ("product", a_vals * b_vals)):
                    rel_error = np.abs(combined - target_vals) / denom
                    match_rate = float((rel_error <= ARITHMETIC_MATCH_TOLERANCE).mean())
                    if match_rate >= ARITHMETIC_MIN_MATCH_RATE and (best is None or match_rate > best["match_rate"]):
                        best = {"method": method, "source_columns": [a, b], "match_rate": round(match_rate, 3)}
        if best:
            findings[target] = best
    return findings


def profiler_node(state: AgenticState) -> dict:
    events = list(state["events"])
    stage_automation = dict(state["stage_automation"])
    try:
        df = state["train_df"]
        target_col = state["target_col"]
        features = df.drop(columns=[target_col])
        non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
        profile = {
            "row_count": len(df),
            "column_count": len(features.columns),
            "dtypes": features.dtypes.astype(str).to_dict(),
            "null_percentages": (features.isnull().mean() * 100).round(2).to_dict(),
            "class_balance": df[target_col].value_counts(normalize=True).round(3).to_dict(),
            "nunique": features.nunique().to_dict(),
            "sample_values_for_non_numeric_columns": {
                col: features[col].dropna().astype(str).unique()[:5].tolist() for col in non_numeric_cols
            },
            "disguised_null_counts_for_non_numeric_columns": {
                col: _count_disguised_nulls(features[col]) for col in non_numeric_cols
            },
            "likely_date_columns": _detect_date_like_columns(features),
            "missingness_correlation_signals": _discover_missingness_correlations(_coerce_effectively_numeric(features)),
            "arithmetic_relationship_signals": _discover_arithmetic_relationships(_coerce_effectively_numeric(features)),
        }
        stage_automation["profiling"] = True
    except Exception as e:
        profile = {"row_count": len(state["train_df"]), "column_count": 0, "dtypes": {}, "null_percentages": {}, "class_balance": {}}
        stage_automation["profiling"] = False
        events.append({"stage": "profiling", "type": "failure", "message": f"{type(e).__name__}: {e}"})
    return {"profile": profile, "stage_automation": stage_automation, "events": events}


DEFAULT_CLEANING_PLAN = {
    "drop_columns": [],
    "treat_as_null_columns": [],
    "coerce_numeric_columns": [],
    "date_columns": {},
    "numeric_impute": "median",
    "categorical_impute": "mode",
    "null_drop_threshold": 0.5,
    "drop_duplicates": True,
    "derived_fills": {},
}

DATE_GRANULARITIES = ("month", "year", "day_of_week", "day")


class DerivedFillRule(BaseModel):
    method: Literal["sum", "product", "zero_when"]
    source_columns: List[str] = []
    when_column: Optional[str] = None
    when_value: Optional[Union[float, str]] = None
    fill_value: Union[float, str] = 0.0


class CleaningPlan(BaseModel):
    drop_columns: List[str] = []
    treat_as_null_columns: List[str] = []
    coerce_numeric_columns: List[str] = []
    date_columns: Dict[str, str] = {}
    numeric_impute: Literal["median", "mean"] = "median"
    categorical_impute: Literal["mode", "constant"] = "mode"
    null_drop_threshold: float = 0.5
    drop_duplicates: bool = True
    derived_fills: Dict[str, DerivedFillRule] = {}


def _reduce_date_column(train_series: pd.Series, test_series: Optional[pd.Series], granularity: str):
    train_dt = pd.to_datetime(train_series, errors="coerce", format="mixed")
    test_dt = pd.to_datetime(test_series, errors="coerce", format="mixed") if test_series is not None else None

    if granularity == "month":
        train_out = train_dt.dt.month
        test_out = test_dt.dt.month if test_dt is not None else None
    elif granularity == "year":
        train_out = train_dt.dt.year
        test_out = test_dt.dt.year if test_dt is not None else None
    elif granularity == "day_of_week":
        train_out = train_dt.dt.dayofweek
        test_out = test_dt.dt.dayofweek if test_dt is not None else None
    else:
        epoch = train_dt.min()
        train_out = (train_dt - epoch).dt.days
        test_out = (test_dt - epoch).dt.days if test_dt is not None else None
    return train_out, test_out


# Each fill uses only that row's other values or a constant, so train and test rows are treated identically with no cross-row leakage.
def _apply_derived_fills(train: pd.DataFrame, test: pd.DataFrame, target_col: str, derived_fills: dict):
    for col, rule in (derived_fills or {}).items():
        if col not in train.columns or col == target_col:
            continue
        method = rule.get("method")
        train_missing = train[col].isnull()
        test_missing = test[col].isnull() if col in test.columns else None

        if method in ("sum", "product"):
            source_cols = [c for c in rule.get("source_columns", []) if c in train.columns]
            if len(source_cols) < 2:
                continue
            combine = (lambda df: df[source_cols].sum(axis=1)) if method == "sum" else (lambda df: df[source_cols].prod(axis=1))
            train.loc[train_missing, col] = combine(train)[train_missing]
            if test_missing is not None and all(c in test.columns for c in source_cols):
                test.loc[test_missing, col] = combine(test)[test_missing]

        elif method == "zero_when":
            when_col = rule.get("when_column")
            when_value = rule.get("when_value")
            fill_value = rule.get("fill_value", 0.0)
            if not when_col or when_col not in train.columns:
                continue
            trigger_train = train_missing & (train[when_col] == when_value)
            train.loc[trigger_train, col] = fill_value
            if test_missing is not None and when_col in test.columns:
                trigger_test = test_missing & (test[when_col] == when_value)
                test.loc[trigger_test, col] = fill_value

    return train, test


# All fill statistics (medians, modes, dropped columns) come from the training frame only.
def _apply_cleaning_plan(train_df, test_df, target_col, positive_label, plan):
    try:
        drop_cols = [c for c in plan.get("drop_columns", []) if c in train_df.columns and c != target_col]
        null_thresh = float(plan.get("null_drop_threshold", 0.5))
        train_null_frac = train_df.drop(columns=[target_col]).isnull().mean()
        high_null_cols = train_null_frac[train_null_frac > null_thresh].index.tolist()
        drop_cols = list(set(drop_cols + high_null_cols))

        train = train_df.drop(columns=drop_cols).copy()
        test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).copy()

        if plan.get("drop_duplicates", True):
            train = train.drop_duplicates()

        disguise_cols = [c for c in plan.get("treat_as_null_columns", []) if c in train.columns and c != target_col]

        def _normalize_disguised_nulls(v):
            if isinstance(v, str) and v.strip().lower() in SENTINEL_MISSING_TOKENS:
                return np.nan
            return v

        for col in disguise_cols:
            train[col] = train[col].apply(_normalize_disguised_nulls)
            if col in test.columns:
                test[col] = test[col].apply(_normalize_disguised_nulls)

        date_plan = plan.get("date_columns", {}) or {}
        for col, granularity in date_plan.items():
            if col not in train.columns or col == target_col:
                continue
            if granularity not in DATE_GRANULARITIES:
                granularity = "month"
            train_out, test_out = _reduce_date_column(
                train[col], test[col] if col in test.columns else None, granularity
            )
            train[col] = train_out
            if test_out is not None:
                test[col] = test_out

        coerce_cols = [c for c in plan.get("coerce_numeric_columns", []) if c in train.columns and c != target_col]
        for col in coerce_cols:
            train[col] = pd.to_numeric(train[col], errors="coerce")
            if col in test.columns:
                test[col] = pd.to_numeric(test[col], errors="coerce")

        train, test = _apply_derived_fills(train, test, target_col, plan.get("derived_fills", {}))

        feature_cols = [c for c in train.columns if c != target_col]
        numeric_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        use_mean = plan.get("numeric_impute") == "mean"
        for col in numeric_cols:
            fill_value = train[col].mean() if use_mean else train[col].median()
            train[col] = train[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)

        use_mode = plan.get("categorical_impute") != "constant"
        for col in categorical_cols:
            if use_mode:
                mode_vals = train[col].mode()
                fill_value = mode_vals.iloc[0] if not mode_vals.empty else "missing"
            else:
                fill_value = "missing"
            train[col] = train[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)

        X_train_clean = train.drop(columns=[target_col])
        X_test_clean = test.drop(columns=[target_col])
        y_train, y_test = encode_target(train[target_col], test[target_col], positive_label)
        return X_train_clean, X_test_clean, y_train, y_test, True

    except Exception:
        X_train_clean = train_df.drop(columns=[target_col]).copy()
        X_test_clean = test_df.drop(columns=[target_col]).copy()
        numeric_cols = X_train_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X_train_clean.columns if c not in numeric_cols]
        for col in numeric_cols:
            fill_value = X_train_clean[col].median()
            X_train_clean[col] = X_train_clean[col].fillna(fill_value)
            X_test_clean[col] = X_test_clean[col].fillna(fill_value)
        for col in categorical_cols:
            mode_vals = X_train_clean[col].mode()
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "missing"
            X_train_clean[col] = X_train_clean[col].fillna(fill_value)
            X_test_clean[col] = X_test_clean[col].fillna(fill_value)
        y_train, y_test = encode_target(train_df[target_col], test_df[target_col], positive_label)
        return X_train_clean, X_test_clean, y_train, y_test, False


def cleaner_node(state: AgenticState) -> dict:
    events = list(state["events"])
    stage_automation = dict(state["stage_automation"])
    token_usage = dict(state["token_usage"])
    used_fallback = state["used_fallback"]

    target_col = state["target_col"]
    train_df = state["train_df"]
    test_df = state["test_df"]

    targeted = _is_targeted(state, "cleaning")
    instruction = state["current_instruction"] if state["weakest_component"] == "cleaning" else None

    if targeted:
        llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
        system_prompt = (
            "You are a data cleaning agent in an automated ML pipeline. Detecting missing "
            "or irregular data is your first job — and missingness is not always a real "
            "NaN. Given a dataset profile, respond with ONLY a JSON object (no prose, no "
            "markdown fences) with these keys: drop_columns (list of column names to drop "
            "entirely, e.g. IDs or near-constant columns), treat_as_null_columns (list of "
            "column names where disguised_null_counts_for_non_numeric_columns in the "
            "profile is greater than zero — those are values that read as missing to a "
            "human, blank/whitespace strings or sentinel tokens like 'N/A', '?', 'unknown', "
            "but are NOT caught by null_percentages, which only counts true NaN; listing a "
            "column here converts those disguised values to real NaN before imputation), "
            "coerce_numeric_columns (list of column names that are "
            "stored as text but actually represent numeric values — check nunique and "
            "sample_values_for_non_numeric_columns in the profile: a column with very high "
            "cardinality relative to row_count, whose sample values look like numbers, is "
            "almost always a numeric field that failed to parse because of a few blank or "
            "malformed entries; one-hot-encoding it verbatim downstream would explode the "
            "feature space into thousands of columns, so catch this here instead), "
            "date_columns (a dict mapping column name to a granularity — 'month', 'year', "
            "'day_of_week', or 'day' — for any column listed under likely_date_columns in "
            "the profile: a full calendar date is often close to one distinct value per "
            "row, exploding a one-hot encoding just as badly as a mistyped numeric column "
            "would; use nunique_month/nunique_year/nunique_day_of_week there to see how "
            "much each granularity actually collapses it, and pick whichever plausibly "
            "carries signal for this problem — e.g. seasonality suggests 'month', weekly "
            "patterns suggest 'day_of_week', long-run trend suggests 'year' or 'day'), "
            "numeric_impute ('median' or 'mean'), "
            "categorical_impute ('mode' or 'constant'), null_drop_threshold (float 0-1, "
            "drop columns with a higher null fraction than this), drop_duplicates (bool), "
            "derived_fills (a dict mapping a missing column's name to a fill rule — see "
            "below). Investigate WHY a column is missing before defaulting to "
            "numeric_impute/categorical_impute for it: the profile's "
            "missingness_correlation_signals shows, for each column with missing values, "
            "which OTHER column's value its missing rows are concentrated at (e.g. a "
            "numeric column sitting at 0, far more often than that value's overall base "
            "rate) — a strong signal the missing value should be a specific constant, not "
            "a generic median. The profile's arithmetic_relationship_signals shows, for "
            "each column with missing values, whether it is closely approximated by the "
            "SUM or PRODUCT of two other numeric columns on the rows where it isn't "
            "missing (match_rate is the fraction of non-missing rows the formula fits) — "
            "if match_rate is high, use that exact formula to derive the missing values "
            "instead of imputing them. A derived_fills rule is one of: "
            "{\"method\": \"product\", \"source_columns\": [colA, colB]} or "
            "{\"method\": \"sum\", \"source_columns\": [colA, colB]} (fills a missing row "
            "from those two OTHER columns' values on that same row — use this when "
            "arithmetic_relationship_signals found a match), or "
            "{\"method\": \"zero_when\", \"when_column\": colX, \"when_value\": v, "
            "\"fill_value\": 0} (fills with fill_value whenever column colX equals v on "
            "that row — use this when missingness_correlation_signals found a "
            "concentration but there's no clean arithmetic formula). when_value can be a "
            "NUMBER (e.g. 0, when colX is numeric — the \"column\"/\"value\" pair "
            "missingness_correlation_signals reports for a numeric driving column) OR a "
            "CATEGORY STRING (e.g. \"Unknown\", when colX is a categorical column — the "
            "same signal reports a string \"value\" for a non-numeric driving column; use "
            "it exactly as given, in quotes, do not coerce it to a number). fill_value can "
            "ALSO be a number or a category string — use a category string when the column "
            "being filled (the one with missing values, not colX) is itself categorical, "
            "e.g. filling a missing smoking_status with 'never smoked' rather than a "
            "number. Only propose a rule "
            "when a signal actually supports it — do not invent a relationship that isn't "
            "in the profile's signals, and do not use derived_fills for a column with no "
            "missing values. Any column not covered by derived_fills still falls back to "
            "numeric_impute/categorical_impute as normal."
        )
        user_prompt = f"Dataset profile:\n{json.dumps(state['profile'], default=str)}"
        if instruction:
            user_prompt += (
                f"\n\nThe reflection agent identified cleaning as the weakest link in the "
                f"previous iteration.\nDiagnosed failure mode: {state['current_failure_mode']}\n"
                f"Instruction: {instruction}\nApply this specific change to your plan."
            )
            events.append({"stage": "cleaning", "type": "instruction_applied", "message": instruction})

        raw, usage, error = call_llm_json(llm, system_prompt, user_prompt)
        token_usage = _add_agent_tokens(token_usage, "cleaning", usage)

        plan = None
        if raw is not None:
            try:
                plan = CleaningPlan.model_validate(raw).model_dump()
            except ValidationError as ve:
                error = f"schema validation failed: {ve}"

        if plan is None:
            events.append({"stage": "cleaning", "type": "fallback", "message": error or "invalid cleaning plan"})
            plan = dict(DEFAULT_CLEANING_PLAN)
            stage_automation["cleaning"] = False
            used_fallback = True
        else:
            stage_automation["cleaning"] = True
    else:
        plan = state["cleaning_plan"]

    X_train_clean, X_test_clean, y_train, y_test, apply_ok = _apply_cleaning_plan(
        train_df, test_df, target_col, state["positive_label"], plan
    )
    if not apply_ok:
        events.append({"stage": "cleaning", "type": "failure", "message": "cleaning plan application failed; used last-resort impute"})
        stage_automation["cleaning"] = False
        used_fallback = True

    return {
        "X_train_clean": X_train_clean, "X_test_clean": X_test_clean,
        "y_train": y_train, "y_test": y_test, "cleaning_plan": plan,
        "stage_automation": stage_automation, "events": events,
        "token_usage": token_usage, "used_fallback": used_fallback,
    }


DEFAULT_FEATURE_PLAN = {
    "categorical_encoding": "label", "scale_numeric": True, "scaler": "standard",
    "numeric_transform": "none", "quantile_bins": None,
    "drop_near_zero_variance": False, "select_top_k_features": None,
}

ORDINAL_ENCODING_FOLDS = 5


class FeaturePlan(BaseModel):
    categorical_encoding: Literal["onehot", "label", "target", "ordinal", "count", "binary"] = "label"
    scale_numeric: bool = True
    scaler: Literal["standard", "minmax", "robust"] = "standard"
    numeric_transform: Literal["none", "log", "yeo_johnson"] = "none"
    quantile_bins: Optional[int] = None
    drop_near_zero_variance: bool = False
    select_top_k_features: Optional[int] = None


# Out-of-fold: each train row is encoded from folds excluding its own, so its label never reaches its own feature; no y_test is accepted.
def _target_encode_column(train_col: pd.Series, test_col: pd.Series, y_train: np.ndarray, seed: int):
    train_vals = train_col.astype(str).to_numpy()
    test_vals = test_col.astype(str).to_numpy()
    y = np.asarray(y_train)
    global_mean = float(y.mean())

    def smoothed_means(categories, targets):
        means = {}
        for cat in np.unique(categories):
            mask = categories == cat
            count = int(mask.sum())
            cat_mean = float(targets[mask].mean())
            means[cat] = (count * cat_mean + TARGET_ENCODING_SMOOTHING * global_mean) / (count + TARGET_ENCODING_SMOOTHING)
        return means

    train_encoded = np.full(len(train_vals), global_mean, dtype=float)
    skf = StratifiedKFold(n_splits=TARGET_ENCODING_FOLDS, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in skf.split(train_vals, y):
        fold_means = smoothed_means(train_vals[fit_idx], y[fit_idx])
        for i in hold_idx:
            train_encoded[i] = fold_means.get(train_vals[i], global_mean)

    full_means = smoothed_means(train_vals, y)
    test_encoded = np.array([full_means.get(v, global_mean) for v in test_vals], dtype=float)
    return train_encoded, test_encoded


# Same out-of-fold scheme as target encoding, but categories are ranked by target rate instead of using the smoothed mean.
def _ordinal_encode_column(train_col: pd.Series, test_col: pd.Series, y_train: np.ndarray, seed: int):
    train_vals = train_col.astype(str).to_numpy()
    test_vals = test_col.astype(str).to_numpy()
    y = np.asarray(y_train)

    def category_ranks(categories, targets):
        means = {cat: float(targets[categories == cat].mean()) for cat in np.unique(categories)}
        ordered = sorted(means, key=lambda c: means[c])
        return {cat: rank for rank, cat in enumerate(ordered)}

    n_categories_full = max(len(np.unique(train_vals)), 1)
    fallback_rank = (n_categories_full - 1) / 2.0

    train_encoded = np.full(len(train_vals), fallback_rank, dtype=float)
    skf = StratifiedKFold(n_splits=ORDINAL_ENCODING_FOLDS, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in skf.split(train_vals, y):
        fold_ranks = category_ranks(train_vals[fit_idx], y[fit_idx])
        fold_fallback = (len(fold_ranks) - 1) / 2.0 if fold_ranks else 0.0
        for i in hold_idx:
            train_encoded[i] = fold_ranks.get(train_vals[i], fold_fallback)

    full_ranks = category_ranks(train_vals, y)
    test_encoded = np.array([full_ranks.get(v, fallback_rank) for v in test_vals], dtype=float)
    return train_encoded, test_encoded


# Counts come from the training column only; an unseen test category gets 0.
def _count_encode_column(train_col: pd.Series, test_col: pd.Series):
    train_vals = train_col.astype(str)
    test_vals = test_col.astype(str)
    counts = train_vals.value_counts()
    train_encoded = train_vals.map(counts).fillna(0).to_numpy(dtype=float)
    test_encoded = test_vals.map(counts).fillna(0).to_numpy(dtype=float)
    return train_encoded, test_encoded


def _binary_encode_columns(X_train_f: pd.DataFrame, X_test_f: pd.DataFrame, categorical_cols: list):
    X_train_f = X_train_f.copy()
    X_test_f = X_test_f.copy()
    for col in categorical_cols:
        categories = X_train_f[col].astype(str).unique().tolist()
        index_map = {cat: i for i, cat in enumerate(categories)}
        n_bits = max(1, int(np.ceil(np.log2(max(len(categories), 2)))))

        def to_bits(v, _index_map=index_map, _n_bits=n_bits):
            idx = _index_map.get(str(v), 0)
            return [int(b) for b in format(idx, f"0{_n_bits}b")]

        train_bits = np.array([to_bits(v) for v in X_train_f[col]])
        test_bits = np.array([to_bits(v) for v in X_test_f[col]])
        bit_cols = [f"{col}_bin{i}" for i in range(n_bits)]
        X_train_f = pd.concat([
            X_train_f.drop(columns=[col]).reset_index(drop=True), pd.DataFrame(train_bits, columns=bit_cols),
        ], axis=1)
        X_test_f = pd.concat([
            X_test_f.drop(columns=[col]).reset_index(drop=True), pd.DataFrame(test_bits, columns=bit_cols),
        ], axis=1)
    return X_train_f, X_test_f


def _apply_numeric_transform(X_train_f: pd.DataFrame, X_test_f: pd.DataFrame, numeric_cols: list, transform: str):
    if not numeric_cols or transform == "none":
        return X_train_f, X_test_f
    if transform == "log":
        safe_cols = [c for c in numeric_cols if (X_train_f[c] >= 0).all()]
        if safe_cols:
            X_train_f[safe_cols] = np.log1p(X_train_f[safe_cols])
            X_test_f[safe_cols] = np.log1p(X_test_f[safe_cols].clip(lower=0))
    elif transform == "yeo_johnson":
        pt = PowerTransformer(method="yeo-johnson")
        X_train_f[numeric_cols] = pt.fit_transform(X_train_f[numeric_cols])
        X_test_f[numeric_cols] = pt.transform(X_test_f[numeric_cols])
    return X_train_f, X_test_f


def _apply_quantile_binning(X_train_f: pd.DataFrame, X_test_f: pd.DataFrame, numeric_cols: list, n_bins: Optional[int]):
    if not numeric_cols or not n_bins:
        return X_train_f, X_test_f, False
    try:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        X_train_f[numeric_cols] = kbd.fit_transform(X_train_f[numeric_cols])
        X_test_f[numeric_cols] = kbd.transform(X_test_f[numeric_cols])
        return X_train_f, X_test_f, True
    except ValueError:
        return X_train_f, X_test_f, False


# Both steps fit on training data only; the selected columns are then applied to test.
def _apply_feature_selection(X_train_f: pd.DataFrame, X_test_f: pd.DataFrame, y_train, plan: dict):
    if plan.get("drop_near_zero_variance"):
        numeric_cols = X_train_f.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            variances = X_train_f[numeric_cols].var()
            drop_cols = variances[variances <= FEATURE_SELECTION_VARIANCE_THRESHOLD].index.tolist()
            if drop_cols:
                X_train_f = X_train_f.drop(columns=drop_cols)
                X_test_f = X_test_f.drop(columns=[c for c in drop_cols if c in X_test_f.columns])

    k = plan.get("select_top_k_features")
    if isinstance(k, int) and 0 < k < X_train_f.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train_f, y_train)
        selected_cols = X_train_f.columns[selector.get_support()]
        X_train_f = X_train_f[selected_cols]
        X_test_f = X_test_f[selected_cols]

    return X_train_f, X_test_f


def _label_encode(X_train_f, X_test_f, categorical_cols):
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_f[col] = le.fit_transform(X_train_f[col].astype(str))
        known = set(le.classes_)
        fallback_class = le.classes_[0]
        X_test_f[col] = X_test_f[col].astype(str).map(lambda v: v if v in known else fallback_class)
        X_test_f[col] = le.transform(X_test_f[col])
    return X_train_f, X_test_f


def _apply_feature_plan(X_train, X_test, y_train, seed, plan):
    try:
        categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_f, X_test_f = X_train.copy(), X_test.copy()

        encoding = plan.get("categorical_encoding")
        # Target/ordinal/count outputs are continuous and get scaled; onehot, label and binary outputs are not.
        continuous_encoded_cols: list = []

        if encoding == "onehot" and categorical_cols:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            train_enc = encoder.fit_transform(X_train_f[categorical_cols])
            test_enc = encoder.transform(X_test_f[categorical_cols])
            enc_cols = encoder.get_feature_names_out(categorical_cols)
            X_train_f = pd.concat([
                X_train_f.drop(columns=categorical_cols).reset_index(drop=True),
                pd.DataFrame(train_enc, columns=enc_cols),
            ], axis=1)
            X_test_f = pd.concat([
                X_test_f.drop(columns=categorical_cols).reset_index(drop=True),
                pd.DataFrame(test_enc, columns=enc_cols),
            ], axis=1)
        elif encoding in ("target", "ordinal") and categorical_cols:
            X_train_f = X_train_f.reset_index(drop=True)
            X_test_f = X_test_f.reset_index(drop=True)
            y_train_arr = np.asarray(y_train)
            encode_fn = _target_encode_column if encoding == "target" else _ordinal_encode_column
            for col in categorical_cols:
                train_encoded, test_encoded = encode_fn(X_train_f[col], X_test_f[col], y_train_arr, seed)
                X_train_f[col] = train_encoded
                X_test_f[col] = test_encoded
            continuous_encoded_cols = list(categorical_cols)
        elif encoding == "count" and categorical_cols:
            for col in categorical_cols:
                train_encoded, test_encoded = _count_encode_column(X_train_f[col], X_test_f[col])
                X_train_f[col] = train_encoded
                X_test_f[col] = test_encoded
            continuous_encoded_cols = list(categorical_cols)
        elif encoding == "binary" and categorical_cols:
            X_train_f, X_test_f = _binary_encode_columns(X_train_f, X_test_f, categorical_cols)
        elif categorical_cols:
            X_train_f, X_test_f = _label_encode(X_train_f, X_test_f, categorical_cols)

        scale_cols = numeric_cols + continuous_encoded_cols
        X_train_f, X_test_f = _apply_numeric_transform(X_train_f, X_test_f, scale_cols, plan.get("numeric_transform", "none"))

        n_bins = plan.get("quantile_bins")
        if n_bins:
            X_train_f, X_test_f, binned = _apply_quantile_binning(X_train_f, X_test_f, scale_cols, n_bins)
        else:
            binned = False
        if not binned and plan.get("scale_numeric", True) and scale_cols:
            scaler = {"minmax": MinMaxScaler(), "robust": RobustScaler()}.get(plan.get("scaler"), StandardScaler())
            X_train_f[scale_cols] = scaler.fit_transform(X_train_f[scale_cols])
            X_test_f[scale_cols] = scaler.transform(X_test_f[scale_cols])

        X_train_f, X_test_f = _apply_feature_selection(X_train_f, X_test_f, y_train, plan)
        return X_train_f, X_test_f, True

    except Exception:
        X_train_f, X_test_f = X_train.copy(), X_test.copy()
        categorical_cols = X_train_f.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_cols = X_train_f.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols:
            X_train_f, X_test_f = _label_encode(X_train_f, X_test_f, categorical_cols)
        if numeric_cols:
            scaler = StandardScaler()
            X_train_f[numeric_cols] = scaler.fit_transform(X_train_f[numeric_cols])
            X_test_f[numeric_cols] = scaler.transform(X_test_f[numeric_cols])
        return X_train_f, X_test_f, False


def feature_engineer_node(state: AgenticState) -> dict:
    events = list(state["events"])
    stage_automation = dict(state["stage_automation"])
    token_usage = dict(state["token_usage"])
    used_fallback = state["used_fallback"]

    X_train = state["X_train_clean"]
    X_test = state["X_test_clean"]

    targeted = _is_targeted(state, "feature_engineering")
    instruction = state["current_instruction"] if state["weakest_component"] == "feature_engineering" else None

    if targeted:
        llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
        system_prompt = (
            "You are a feature engineering agent in an automated ML pipeline. Respond "
            "with ONLY a JSON object (no prose, no markdown fences) with keys: "
            "categorical_encoding — a SINGLE string, one of 'onehot', 'label', 'target', "
            "'ordinal', 'count', or 'binary', applied uniformly to every categorical "
            "column this round (NOT a per-column mapping/dict — pick the one encoding "
            "that best fits the overall cardinality profile below, even if columns "
            "differ somewhat), scale_numeric (bool), scaler ('standard', 'minmax', or 'robust'), "
            "numeric_transform ('none', 'log', or 'yeo_johnson'), quantile_bins (an "
            "integer to discretize numeric columns into that many quantile buckets "
            "INSTEAD of scaling, or null to leave scaling as-is), drop_near_zero_variance "
            "(bool), select_top_k_features (an integer, or null to keep all features).\n\n"
            "Categorical encoding — base the choice on the actual per-column "
            "cardinalities given below, not a generic default:\n"
            "'onehot': adds (cardinality - 1) new columns for that column alone — fine "
            "for low cardinality, blows up the feature space for high cardinality.\n"
            "'label': one column, arbitrary integer per category — can mislead a linear "
            "model into treating unrelated categories as ordered.\n"
            "'target': one column, smoothed mean-of-target — dense, no dimensionality "
            "explosion regardless of cardinality, computed out-of-fold so it never leaks "
            "the label. Prefer it for a high-cardinality column.\n"
            "'ordinal': one column, but genuinely different from 'label' — categories are "
            "RANKED by their out-of-fold relationship to the target (lowest positive rate "
            "-> 0, highest -> k-1), also computed out-of-fold so it never leaks the label. "
            "A middle ground: carries real signal like target encoding, but only a rank, "
            "not the full mean.\n"
            "'count': one column, each category replaced by its training-set frequency — "
            "does not touch the target at all, useful when how COMMON a category is (not "
            "its relationship to the outcome) is the informative part.\n"
            "'binary': ceil(log2(cardinality)) new columns (each category's index in "
            "binary) — a compromise between one-hot's dimensionality and label's single "
            "arbitrary column, does not touch the target.\n\n"
            "Numeric transform (applied before scaling): 'log' for right-skewed, "
            "non-negative columns (e.g. monetary amounts); 'yeo_johnson' for skewed "
            "columns that may include negative or zero values (handles them natively, "
            "unlike log). quantile_bins is an alternative to scaling, not an addition to "
            "it — use it when a numeric column's raw scale is less informative than which "
            "quantile bucket a value falls into.\n\n"
            "Use drop_near_zero_variance to remove features carrying essentially no "
            "signal, and select_top_k_features to keep only the k most informative "
            "features (by a univariate test) when the feature count is large relative to "
            "row_count."
        )
        categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cardinality = {c: int(X_train[c].nunique()) for c in categorical_cols}
        user_prompt = (
            f"Cleaned training data has {len(numeric_cols)} numeric columns and "
            f"{len(categorical_cols)} categorical columns, {len(X_train)} rows. "
            f"Per-column cardinality of the categorical columns: {categorical_cardinality}."
        )
        if instruction:
            user_prompt += (
                f"\n\nThe reflection agent identified feature engineering as the weakest "
                f"link in the previous iteration.\nDiagnosed failure mode: "
                f"{state['current_failure_mode']}\nInstruction: {instruction}\n"
                f"Apply this specific change to your plan."
            )
            events.append({"stage": "feature_engineering", "type": "instruction_applied", "message": instruction})

        raw, usage, error = call_llm_json(llm, system_prompt, user_prompt)
        token_usage = _add_agent_tokens(token_usage, "feature_engineering", usage)

        plan = None
        if raw is not None:
            try:
                plan = FeaturePlan.model_validate(raw).model_dump()
            except ValidationError as ve:
                error = f"schema validation failed: {ve}"

        if plan is None:
            events.append({"stage": "feature_engineering", "type": "fallback", "message": error or "invalid feature plan"})
            plan = dict(DEFAULT_FEATURE_PLAN)
            stage_automation["feature_engineering"] = False
            used_fallback = True
        else:
            stage_automation["feature_engineering"] = True
    else:
        plan = state["feature_plan"]

    X_train_f, X_test_f, apply_ok = _apply_feature_plan(X_train, X_test, state["y_train"], state["seed"], plan)
    if not apply_ok:
        events.append({"stage": "feature_engineering", "type": "failure", "message": "feature plan application failed; used label-encode+standard-scale fallback"})
        stage_automation["feature_engineering"] = False
        used_fallback = True

    return {
        "X_train_features": X_train_f, "X_test_features": X_test_f, "feature_plan": plan,
        "stage_automation": stage_automation, "events": events,
        "token_usage": token_usage, "used_fallback": used_fallback,
    }


def model_selector_node(state: AgenticState) -> dict:
    events = list(state["events"])
    stage_automation = dict(state["stage_automation"])
    token_usage = dict(state["token_usage"])
    used_fallback = state["used_fallback"]
    tried_models = list(state.get("tried_models", []))

    # initial_pass: nothing decided yet. model_reselect: re-decide the family. upstream_retune: keep the family, re-tune on the changed features.
    if state["weakest_component"] is None:
        tuning_case = "initial_pass"
    elif state["weakest_component"] == "model_selection":
        tuning_case = "model_reselect"
    else:
        tuning_case = "upstream_retune"

    decide_family = tuning_case in ("initial_pass", "model_reselect")
    instruction = state["current_instruction"] if tuning_case == "model_reselect" else None

    if decide_family:
        llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
        system_prompt = (
            "You are the model selection agent in an automated ML pipeline. You choose "
            f"the model family; its hyperparameters are then tuned for you by Optuna. "
            f"Choose one model from exactly this list: {CANDIDATE_MODELS}. Respond with "
            "ONLY a JSON object (no prose, no markdown fences) with keys: model (one of "
            "the listed names), use_class_weighting (bool), reason (a short string).\n\n"
            "use_class_weighting: set true to make the model account for class imbalance "
            "during training — logistic_regression, random_forest, extra_trees, svm, and "
            "decision_tree get class_weight='balanced'; xgboost and lightgbm get "
            "scale_pos_weight set to the actual negative:positive ratio in the training "
            "data; catboost gets auto_class_weights='Balanced'. gradient_boosting, knn, "
            "naive_bayes, and adaboost have no such option in this pipeline — setting it "
            "true for one of those is simply a no-op, so prefer a model that supports it "
            "when imbalance is the real problem. The minority-class rate for THIS dataset "
            "is given below — weight the decision by it: class weighting has its largest "
            "effect on severely imbalanced data (roughly under 10% minority, e.g. a "
            "medical-screening target), a moderate effect in the ~10-20% range, and "
            "progressively less as the classes approach balance — near 25-35% minority "
            "the classes are only mildly imbalanced and class weighting mostly just "
            "trades precision for a little recall without a clear net gain, so the "
            "default there should be false unless a prior iteration's minority-class "
            "recall was specifically poor. Don't set it true reflexively just because "
            "the classes aren't exactly 50/50."
        )
        already_tried = f" Already tried this run: {tried_models}." if tried_models else ""
        class_balance = state["profile"].get("class_balance") or {}
        minority_rate = min(class_balance.values()) if class_balance else None
        minority_str = (
            f"Minority-class rate: {minority_rate:.1%} "
            f"({'severely imbalanced' if minority_rate < 0.10 else 'moderately imbalanced' if minority_rate < 0.20 else 'mildly imbalanced' if minority_rate < 0.40 else 'roughly balanced'})."
            if minority_rate is not None else ""
        )
        user_prompt = (
            f"Training data: {len(state['X_train_features'])} rows, "
            f"{state['X_train_features'].shape[1]} features. "
            f"Class balance: {class_balance}. {minority_str}{already_tried} "
            "Pick the most promising model, preferring one not already tried."
        )
        if instruction:
            user_prompt += (
                f"\n\nThe reflection agent identified model selection as the weakest link "
                f"in the previous iteration.\nDiagnosed failure mode: "
                f"{state['current_failure_mode']}\nInstruction: {instruction}\n"
                f"Apply this specific change."
            )
            events.append({"stage": "model_selection", "type": "instruction_applied", "message": instruction})

        plan, usage, error = call_llm_json(llm, system_prompt, user_prompt)
        token_usage = _add_agent_tokens(token_usage, "model_selection", usage)

        model_name = plan.get("model") if plan else None
        if model_name not in CANDIDATE_MODELS:
            events.append({"stage": "model_selection", "type": "fallback", "message": error or f"invalid model choice {model_name!r}"})
            remaining = [m for m in CANDIDATE_MODELS if m not in tried_models]
            model_name = remaining[0] if remaining else CANDIDATE_MODELS[0]
            stage_automation["model_selection"] = False
            used_fallback = True
        else:
            stage_automation["model_selection"] = True
        use_class_weighting = bool(plan.get("use_class_weighting")) if plan else False
        tried_models = tried_models + [model_name]
        default_trials = OPTUNA_TRIALS_FIRST
    else:
        model_name = state["model_name"]
        use_class_weighting = state.get("use_class_weighting", False)
        default_trials = OPTUNA_TRIALS_RETUNE

    n_trials = state["optuna_trials"] if state["optuna_trials"] is not None else default_trials

    # Tuning may use a subsample; the final fit always uses the full training set.
    X_full, y_full = state["X_train_features"], state["y_train"]
    used_subsample = len(X_full) > OPTUNA_SUBSAMPLE_ROW_THRESHOLD
    if used_subsample:
        X_tune, _, y_tune, _ = train_test_split(
            X_full, y_full, train_size=OPTUNA_SUBSAMPLE_SIZE, stratify=y_full, random_state=state["seed"],
        )
    else:
        X_tune, y_tune = X_full, y_full

    # Computed once from training labels only and reused for tuning and the final fit.
    class_weight_ratio = None
    if use_class_weighting:
        pos = int((np.asarray(y_full) == 1).sum())
        neg = int((np.asarray(y_full) == 0).sum())
        class_weight_ratio = neg / max(pos, 1)

    t_opt0 = time.perf_counter()
    best_params, tuning_ok = _run_optuna(X_tune, y_tune, model_name, state["seed"], n_trials, class_weight_ratio)
    optuna_durations = list(state["optuna_durations"]) + [round(time.perf_counter() - t_opt0, 4)]
    stage_automation["hyperparameter_tuning"] = tuning_ok
    if not tuning_ok:
        events.append({"stage": "hyperparameter_tuning", "type": "fallback", "message": "Optuna study failed; used default hyperparameters"})
        used_fallback = True

    events.append({
        "stage": "model_selection", "type": "tuning_case",
        "message": f"{tuning_case}: model={model_name}, n_trials={n_trials}, subsampled={used_subsample}, "
                   f"class_weighting={use_class_weighting}",
    })

    model = build_model(model_name, best_params, state["seed"], class_weight_ratio)
    t0 = time.perf_counter()
    model.fit(X_full, y_full)
    train_time = time.perf_counter() - t0

    return {
        "model_name": model_name, "best_params": best_params, "tried_models": tried_models,
        "model": model, "train_time_sec": round(train_time, 4), "optuna_durations": optuna_durations,
        "tuning_case": tuning_case, "use_class_weighting": use_class_weighting, "class_weight_ratio": class_weight_ratio,
        "stage_automation": stage_automation, "events": events,
        "token_usage": token_usage, "used_fallback": used_fallback,
    }


def evaluator_node(state: AgenticState) -> dict:
    events = list(state["events"])
    stage_automation = dict(state["stage_automation"])
    threshold = 0.5
    try:
        model = state["model"]
        X_train, y_train = state["X_train_features"], state["y_train"]
        X_test, y_test = state["X_test_features"], state["y_test"]
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        if y_proba is not None:
            # Threshold is calibrated out-of-fold on training data with a fresh clone using the same class weighting; the fitted model only predicts on X_test.
            calibration_model = build_model(state["model_name"], state["best_params"], state["seed"], state.get("class_weight_ratio"))
            threshold = calibrate_threshold_oof(
                calibration_model, X_train, y_train, state["seed"],
                metric=state.get("calibration_metric", THRESHOLD_CALIBRATION_METRIC),
                subsample_row_threshold=OPTUNA_SUBSAMPLE_ROW_THRESHOLD, subsample_size=OPTUNA_SUBSAMPLE_SIZE,
            )
            y_pred = (y_proba[:, 1] >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba)

        train_auc = None
        if y_proba is not None:
            try:
                train_proba = model.predict_proba(X_train)[:, 1]
                train_auc = round(float(roc_auc_score(y_train, train_proba)), 4)
            except ValueError:
                train_auc = None
        test_auc = metrics.get("auc_roc")
        gap = round(train_auc - test_auc, 4) if (train_auc is not None and test_auc is not None) else None
        metrics["train_auc_roc"] = train_auc
        metrics["train_test_gap"] = gap
        metrics["overfitting_flag"] = bool(gap is not None and gap > OVERFITTING_GAP_THRESHOLD)
        metrics["underfitting_flag"] = bool(
            train_auc is not None and test_auc is not None
            and train_auc < UNDERFITTING_AUC_THRESHOLD and test_auc < UNDERFITTING_AUC_THRESHOLD
        )

        stage_automation["evaluation"] = True
    except Exception as e:
        metrics = {
            "accuracy": None, "f1_weighted": None, "auc_roc": None, "recall_pos": None, "precision_pos": None,
            "train_auc_roc": None, "train_test_gap": None, "overfitting_flag": False, "underfitting_flag": False,
        }
        stage_automation["evaluation"] = False
        events.append({"stage": "evaluation", "type": "failure", "message": f"{type(e).__name__}: {e}"})
    return {"metrics": metrics, "threshold": threshold, "stage_automation": stage_automation, "events": events}


class ReflectionDiagnosis(BaseModel):
    weakest_component: Literal["cleaning", "feature_engineering", "model_selection"]
    failure_mode: str
    instruction: str
    calibration_metric: Literal["f1", "f2"] = "f1"


def _get_reflection_diagnosis(llm: ChatAnthropic, system_prompt: str, user_prompt: str):
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    last_error = None
    for attempt in range(2):
        raw, usage, error = call_llm_json(llm, system_prompt, user_prompt)
        total_usage = _sum_usage(total_usage, usage)
        if raw is not None:
            try:
                return ReflectionDiagnosis.model_validate(raw), total_usage, False, None
            except ValidationError as ve:
                last_error = f"schema validation failed (attempt {attempt + 1}): {ve}"
        else:
            last_error = f"{error} (attempt {attempt + 1})"

    fallback = ReflectionDiagnosis(
        weakest_component="model_selection",
        failure_mode="reflection response was invalid JSON or failed schema validation on both attempts",
        instruction="try a different model family and widen the Optuna hyperparameter search",
    )
    return fallback, total_usage, True, last_error


def reflection_node(state: AgenticState) -> dict:
    events = list(state["events"])
    token_usage = dict(state["token_usage"])
    used_fallback = state["used_fallback"]
    iteration = state["iteration"] + 1
    current_metrics = state["metrics"] or {}

    features_df = state.get("X_train_features")
    history_entry = {
        "iteration": iteration,
        "acted_on_component": state["weakest_component"],
        "acted_on_instruction": state["current_instruction"],
        "cleaning_plan": state["cleaning_plan"],
        "feature_plan": state["feature_plan"],
        "final_feature_count": int(features_df.shape[1]) if features_df is not None else None,
        "model": state["model_name"],
        "best_params": state["best_params"],
        "tuning_case": state.get("tuning_case"),
        "threshold": state.get("threshold"),
        "use_class_weighting": state.get("use_class_weighting"),
        "class_weight_ratio": state.get("class_weight_ratio"),
        "calibration_metric_used": state.get("calibration_metric"),
        **current_metrics,
    }
    history = list(state["iteration_history"]) + [history_entry]

    llm = ChatAnthropic(model=REFLECTION_LLM_MODEL, temperature=0)
    system_prompt = (
        "You are the reflection agent in an automated ML pipeline. You are the expert data scientist reviewing the junior's work process. After each "
        "iteration you analyze the result, identify the single weakest "
        "component, diagnose why it is limiting performance, and give "
        "concrete instructions for improving it next round. Ground your "
        "diagnosis in the actual numbers below (e.g. final_feature_count, the "
        "cleaning_plan and feature_plan that were actually used) rather than "
        "generic best-practice assumptions — if final_feature_count is small, "
        "for example, 'one-hot encoding creates a sparse high-dimensional "
        "space' is not an accurate diagnosis for this run, whatever it might "
        "usually be true of in general.\n\n"
        "The junior now has a much wider toolkit than before — phrase your "
        "instruction in these exact terms when you want it used, or the "
        "schema can't act on a vaguer suggestion:\n"
        "cleaning: derived_fills is not just impute-or-drop — each iteration's "
        "cleaning_plan and the run's profile carry missingness_correlation_signals "
        "(a column's missing rows concentrated at some other column's fixed "
        "value) and arithmetic_relationship_signals (a column closely equal "
        "to the sum or product of two others). If a column with missing "
        "values still has numeric_impute/categorical_impute applied instead "
        "of a derived_fills rule, and the signals actually support one, name "
        "the exact rule to add (e.g. 'add a derived_fills product rule for "
        "TotalCharges from tenure and MonthlyCharges'). Never instruct a "
        "derived_fills rule the profile's own signals don't support.\n"
        "feature_engineering: categorical_encoding also has 'ordinal' (ranks "
        "categories by their out-of-fold relationship to the target — a "
        "middle ground between label's arbitrary order and target's full "
        "mean), 'count' (frequency in training data, no target involved), "
        "and 'binary' (log2(cardinality) columns, no target involved) — on "
        "top of the existing 'onehot'/'label'/'target'. numeric_transform "
        "('log' for skewed non-negative columns, 'yeo_johnson' for skewed "
        "columns that may be negative) and quantile_bins (discretize instead "
        "of scale) are also real, implemented options.\n"
        "model_selection: the model pool is large now — logistic_regression, "
        "random_forest, gradient_boosting, extra_trees, svm, knn, "
        "naive_bayes, decision_tree, adaboost, xgboost, lightgbm, catboost. "
        "Use the run's overfitting_flag/underfitting_flag (see below) to "
        "pick a direction: overfitting suggests a simpler/more regularized "
        "model (e.g. logistic_regression, naive_bayes, a shallower tree) or "
        "fewer features; underfitting suggests a more expressive model (e.g. "
        "xgboost, lightgbm, catboost) or richer features, not just 'try a "
        "different model' with no direction. It can also set "
        "use_class_weighting=true (logistic_regression/random_forest/"
        "extra_trees/svm/decision_tree get class_weight='balanced'; xgboost/"
        "lightgbm get scale_pos_weight; catboost gets auto_class_weights) — "
        "if the class balance is imbalanced (a minority class under ~20%) "
        "and recall_pos or precision_pos on the minority class looks weak, "
        "instruct model_selection to turn this on. It's a training-time-only "
        "change (no test data involved) and a no-op for gradient_boosting/"
        "knn/naive_bayes/adaboost, which don't support it.\n\n"
        "Every iteration's entry below also carries train_auc_roc, "
        "train_test_gap (train minus test AUC-ROC), overfitting_flag, and "
        "underfitting_flag — computed independently of the calibrated "
        "threshold, from raw predicted probabilities. A large positive gap "
        "means the model is memorizing training data rather than "
        "generalizing; both train and test AUC-ROC being low together means "
        "the model or features are too weak to separate the classes at all, "
        "which is a different problem from overfitting and calls for a "
        "different instruction (richer features or a more expressive model, "
        "not more regularization). Use these signals — don't diagnose "
        "overfitting from test performance alone, that's exactly what they "
        "exist to make visible.\n\n"
        "You also choose calibration_metric ('f1' or 'f2') every round, "
        "independent of weakest_component — this controls which objective "
        "the evaluator's out-of-fold threshold search optimizes next "
        "iteration. 'f1' (the default) balances precision and recall "
        "equally. 'f2' weights recall roughly twice as heavily as "
        "precision — prefer it for a screening-style problem where missing "
        "a real positive case is worse than a false alarm (e.g. Stroke: "
        "missing an actual stroke case is worse than flagging a healthy "
        "patient for review), especially if recall_pos looks low relative "
        "to what the model's AUC-ROC suggests it should be able to achieve. "
        "This choice does not change how the threshold search itself works "
        "(still out-of-fold, still training data only) — only which score "
        "it's searching for. IMPORTANT: 'f2' and model-level class weighting "
        "(use_class_weighting in the model plan) are two independent "
        "corrections for the SAME problem — class weighting already shifts "
        "the model's decision boundary toward the minority class, and 'f2' "
        "then shifts the threshold further the same way. Stacking both on "
        "MILDLY imbalanced data (minority class above ~20%) tends to "
        "over-correct: it can push recall very high while collapsing "
        "precision and accuracy (this happened on Credit Default — one run "
        "hit recall 0.89 but precision 0.30 and accuracy 0.51). If the "
        "model plan already has class weighting on and the data is only "
        "mildly imbalanced, keeping calibration_metric at 'f1' is usually "
        "the more stable choice; reserve 'f2'-on-top-of-weighting for "
        "genuinely severe imbalance (e.g. Stroke's ~5%).\n\n"
        "Aim your instruction at whatever change is most likely to move "
        "AUC-ROC toward 0.85 — but a target isn't always reachable from "
        "where a run stands, so stay grounded in the real numbers rather "
        "than promising it will be hit. A bigger toolkit does not mean "
        "every change helps — if the previous round's instruction made "
        "things worse (compare this iteration's metrics to the one before "
        "it), say so plainly and try something different, don't repeat it. "
        "Respond with "
        "ONLY a JSON object (no prose, no markdown fences) with exactly these "
        "keys: weakest_component (one of 'cleaning', 'feature_engineering', "
        "'model_selection'), failure_mode (a short diagnosis string), "
        "instruction (one concrete, specific change for that component only), "
        "calibration_metric ('f1' or 'f2')."
    )
    user_prompt = f"Iteration history so far:\n{json.dumps(history, default=str)}"

    diagnosis, usage, fell_back, error = _get_reflection_diagnosis(llm, system_prompt, user_prompt)
    token_usage = _add_agent_tokens(token_usage, "reflection", usage)
    if fell_back:
        events.append({"stage": "reflection", "type": "fallback", "message": error or "reflection fallback"})
        used_fallback = True

    history[-1]["diagnosis"] = diagnosis.model_dump()

    metric_now = current_metrics.get(PERFORMANCE_TARGET_METRIC)
    metric_prev = history[-2].get(PERFORMANCE_TARGET_METRIC) if len(history) >= 2 else None

    did_not_improve = (
        metric_prev is not None
        and metric_now is not None
        and (metric_now - metric_prev) <= state["no_improvement_epsilon"]
    )
    no_improvement_streak = (state["no_improvement_streak"] + 1) if did_not_improve else 0
    history[-1]["no_improvement_streak"] = no_improvement_streak

    stop_reason = None
    if (
        state["performance_target"] is not None
        and metric_now is not None
        and metric_now >= state["performance_target"]
    ):
        stop_reason = "performance_target_reached"
    elif iteration >= state["max_iterations"]:
        stop_reason = "iteration_cap_reached"
    elif no_improvement_streak >= NO_IMPROVEMENT_PATIENCE:
        stop_reason = "no_improvement"

    history[-1]["stop_reason"] = stop_reason

    return {
        "iteration": iteration,
        "iteration_history": history,
        "weakest_component": diagnosis.weakest_component,
        "current_instruction": diagnosis.instruction,
        "current_failure_mode": diagnosis.failure_mode,
        "calibration_metric": diagnosis.calibration_metric,
        "stop_reason": stop_reason,
        "no_improvement_streak": no_improvement_streak,
        "events": events,
        "token_usage": token_usage,
        "used_fallback": used_fallback,
    }


def reflection_router(state: AgenticState) -> str:
    return "end" if state["stop_reason"] else "continue"


def _timed(stage_name: str, node_fn):
    def wrapper(state: AgenticState) -> dict:
        iter_num = state.get("iteration", 0) + 1
        print(f"[iter {iter_num}] {stage_name}: starting...", flush=True)
        t0 = time.perf_counter()
        result = node_fn(state)
        dt = round(time.perf_counter() - t0, 4)
        print(f"[iter {iter_num}] {stage_name}: done in {dt}s", flush=True)
        timings = list(state.get("stage_timings", [])) + [{"stage": stage_name, "duration_sec": dt}]
        result = dict(result)
        result["stage_timings"] = timings
        return result
    return wrapper


def build_pipeline():
    graph = StateGraph(AgenticState)
    graph.add_node("profiler", _timed("profiler", profiler_node))
    graph.add_node("cleaner", _timed("cleaner", cleaner_node))
    graph.add_node("feature_engineer", _timed("feature_engineer", feature_engineer_node))
    graph.add_node("model_selector", _timed("model_selector", model_selector_node))
    graph.add_node("evaluator", _timed("evaluator", evaluator_node))
    graph.add_node("reflection", _timed("reflection", reflection_node))

    graph.set_entry_point("profiler")
    graph.add_edge("profiler", "cleaner")
    graph.add_edge("cleaner", "feature_engineer")
    graph.add_edge("feature_engineer", "model_selector")
    graph.add_edge("model_selector", "evaluator")
    graph.add_edge("evaluator", "reflection")
    # Always loops back to the cleaner; each node decides whether to re-decide or replay its stored plan.
    graph.add_conditional_edges("reflection", reflection_router, {
        "end": END,
        "continue": "cleaner",
    })
    return graph.compile()


def _build_initial_state(
    train_df, test_df, target_col, positive_label, dataset_name, seed,
    dry_run, max_iterations, optuna_trials, performance_target, no_improvement_epsilon,
) -> AgenticState:
    return {
        "train_df": train_df, "test_df": test_df, "target_col": target_col,
        "positive_label": positive_label, "dataset_name": dataset_name, "seed": seed,
        "max_iterations": max_iterations, "optuna_trials": optuna_trials, "dry_run": dry_run,
        "performance_target": performance_target, "no_improvement_epsilon": no_improvement_epsilon,
        "profile": {},
        "cleaning_plan": None, "feature_plan": None,
        "X_train_clean": None, "X_test_clean": None,
        "y_train": None, "y_test": None, "X_train_features": None, "X_test_features": None,
        "model_name": None, "model": None, "best_params": None, "train_time_sec": None,
        "tuning_case": None, "use_class_weighting": False, "class_weight_ratio": None,
        "metrics": None, "threshold": None, "calibration_metric": THRESHOLD_CALIBRATION_METRIC,
        "iteration": 0, "iteration_history": [],
        "stage_automation": {s: False for s in SIX_STAGES}, "events": [],
        "used_fallback": False, "token_usage": {},
        "stage_timings": [], "optuna_durations": [],
        "weakest_component": None, "current_instruction": None, "current_failure_mode": None,
        "stop_reason": None, "no_improvement_streak": 0, "tried_models": [],
    }


# Unlike f1_weighted, collapses when minority-class precision or recall does; returns -1.0 if either is missing.
def _positive_class_f1(entry: dict) -> float:
    p, r = entry.get("precision_pos"), entry.get("recall_pos")
    if p is None or r is None or (p + r) == 0:
        return -1.0
    return 2 * p * r / (p + r)


# Among iterations within BEST_ITERATION_AUC_TOLERANCE of the best AUC-ROC, pick the highest positive-class F1 (earliest on ties).
def _select_best_iteration(scored: list) -> Optional[dict]:
    if not scored:
        return None
    best_auc = max(h[PERFORMANCE_TARGET_METRIC] for h in scored)
    # 1e-9 absorbs floating-point error at the tolerance boundary.
    tied = [h for h in scored if best_auc - h[PERFORMANCE_TARGET_METRIC] <= BEST_ITERATION_AUC_TOLERANCE + 1e-9]
    return max(tied, key=_positive_class_f1)


# Reports the iteration chosen by _select_best_iteration, not necessarily the last one.
def build_result_from_state(final_state: AgenticState, dataset_name, seed, max_iterations, optuna_trials, performance_target) -> RunResult:
    result = RunResult(system="agentic", dataset=dataset_name, seed=seed, success=False)
    try:
        history = final_state["iteration_history"]

        scored = [h for h in history if h.get(PERFORMANCE_TARGET_METRIC) is not None]
        best_entry = _select_best_iteration(scored)
        for h in history:
            h["selected_as_final"] = h is best_entry

        if best_entry is not None:
            metrics = {k: best_entry.get(k) for k in
                       ("accuracy", "f1_weighted", "auc_roc", "recall_pos", "precision_pos",
                        "train_auc_roc", "train_test_gap", "overfitting_flag", "underfitting_flag")}
            best_model_name = best_entry.get("model")
            best_iteration_num = best_entry.get("iteration")
            best_threshold = best_entry.get("threshold")
            best_use_class_weighting = best_entry.get("use_class_weighting")
            best_class_weight_ratio = best_entry.get("class_weight_ratio")
            best_calibration_metric = best_entry.get("calibration_metric_used")
        else:
            metrics = final_state["metrics"] or {}
            best_model_name = final_state.get("model_name")
            best_iteration_num = final_state.get("iteration")
            best_threshold = final_state.get("threshold")
            best_use_class_weighting = final_state.get("use_class_weighting")
            best_class_weight_ratio = final_state.get("class_weight_ratio")
            best_calibration_metric = final_state.get("calibration_metric")

        result.accuracy = metrics.get("accuracy")
        result.f1_weighted = metrics.get("f1_weighted")
        result.auc_roc = metrics.get("auc_roc")
        result.recall_pos = metrics.get("recall_pos")
        result.precision_pos = metrics.get("precision_pos")
        result.threshold_used = best_threshold
        result.model_train_time_sec = final_state.get("train_time_sec")
        best_train_auc = metrics.get("train_auc_roc")
        best_gap = metrics.get("train_test_gap")
        result.stage_automation = final_state["stage_automation"]
        result.events = final_state["events"]
        result.used_fallback = final_state["used_fallback"]
        result.iteration_count = final_state["iteration"]
        result.iteration_metrics = history

        per_agent_usage = final_state["token_usage"]
        per_agent_cost = {agent: round(_token_cost_usd(usage), 6) for agent, usage in per_agent_usage.items()}
        result.api_token_cost_usd = round(sum(per_agent_cost.values()), 6)

        stage_durations: Dict[str, float] = {}
        for entry in final_state["stage_timings"]:
            stage_durations[entry["stage"]] = round(stage_durations.get(entry["stage"], 0.0) + entry["duration_sec"], 4)
        optuna_total = round(sum(final_state["optuna_durations"]), 4)

        result.extra_params = {
            "final_model": best_model_name,
            "best_iteration": best_iteration_num,
            "last_iteration": final_state.get("iteration"),
            "train_auc_roc": best_train_auc,
            "train_test_gap": best_gap,
            "overfitting_flag": metrics.get("overfitting_flag"),
            "underfitting_flag": metrics.get("underfitting_flag"),
            "use_class_weighting": best_use_class_weighting,
            "class_weight_ratio": best_class_weight_ratio,
            "calibration_metric": best_calibration_metric,
            "max_iterations": max_iterations,
            "optuna_trials_override": optuna_trials,
            "optuna_trials_first": optuna_trials if optuna_trials is not None else OPTUNA_TRIALS_FIRST,
            "optuna_trials_retune": optuna_trials if optuna_trials is not None else OPTUNA_TRIALS_RETUNE,
            "performance_target": performance_target,
            "stop_reason": final_state.get("stop_reason"),
            "stage_wallclock_sec": stage_durations,
            "optuna_wallclock_sec_total": optuna_total,
            "optuna_wallclock_sec_per_call": final_state["optuna_durations"],
            "token_usage_by_agent": per_agent_usage,
            "cost_by_agent_usd": per_agent_cost,
        }
        result.success = result.accuracy is not None
        if not result.success:
            result.failure_cause = "evaluation stage did not produce metrics"
    except Exception as e:
        result.failure_cause = f"{type(e).__name__}: {e}"
        result.events.append({"stage": "orchestration", "type": "failure", "message": traceback.format_exc()})
    return result


def run_agentic(
    train_df, test_df, target_col, positive_label, dataset_name, seed,
    dry_run: bool = False,
    max_iterations: Optional[int] = None,
    optuna_trials: Optional[int] = None,
    performance_target: Optional[float] = PERFORMANCE_TARGET,
    no_improvement_epsilon: float = NO_IMPROVEMENT_EPSILON,
) -> RunResult:
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS_DRY_RUN if dry_run else MAX_ITERATIONS_FULL
    if dry_run:
        performance_target = None

    pipeline = build_pipeline()
    initial_state = _build_initial_state(
        train_df, test_df, target_col, positive_label, dataset_name, seed,
        dry_run, max_iterations, optuna_trials, performance_target, no_improvement_epsilon,
    )
    try:
        final_state = pipeline.invoke(initial_state, config={"recursion_limit": 50})
    except Exception as e:
        result = RunResult(system="agentic", dataset=dataset_name, seed=seed, success=False)
        result.failure_cause = f"{type(e).__name__}: {e}"
        result.events.append({"stage": "orchestration", "type": "failure", "message": traceback.format_exc()})
        return result

    return build_result_from_state(final_state, dataset_name, seed, max_iterations, optuna_trials, performance_target)


def run_agentic_streaming(
    train_df, test_df, target_col, positive_label, dataset_name, seed,
    dry_run: bool = False,
    max_iterations: Optional[int] = None,
    optuna_trials: Optional[int] = None,
    performance_target: Optional[float] = PERFORMANCE_TARGET,
    no_improvement_epsilon: float = NO_IMPROVEMENT_EPSILON,
):
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS_DRY_RUN if dry_run else MAX_ITERATIONS_FULL
    if dry_run:
        performance_target = None

    pipeline = build_pipeline()
    initial_state = _build_initial_state(
        train_df, test_df, target_col, positive_label, dataset_name, seed,
        dry_run, max_iterations, optuna_trials, performance_target, no_improvement_epsilon,
    )
    yield from pipeline.stream(initial_state, config={"recursion_limit": 50}, stream_mode="values")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the six-agent agentic pipeline standalone.")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=None, help=f"Override the Optuna trial budget for every case (default: 2 under --dry-run, else adaptive — {OPTUNA_TRIALS_FIRST} first-tune / {OPTUNA_TRIALS_RETUNE} re-tune).")
    args = parser.parse_args()

    ensure_api_key()
    optuna_trials = args.optuna_trials if args.optuna_trials is not None else (2 if args.dry_run else None)

    df, target_col = load_dataset(args.dataset)
    positive_label = DATASETS[args.dataset]["positive_label"]
    train_df, test_df = make_split(df, target_col, args.seed)

    print(
        f"Starting: dataset={args.dataset} seed={args.seed} "
        f"optuna_trials={optuna_trials if optuna_trials is not None else f'adaptive ({OPTUNA_TRIALS_FIRST} first / {OPTUNA_TRIALS_RETUNE} retune)'} "
        f"max_iterations={'2 (dry-run)' if args.dry_run else '5'} "
        f"performance_target={'disabled (dry-run)' if args.dry_run else PERFORMANCE_TARGET} "
        f"no_improvement_patience={NO_IMPROVEMENT_PATIENCE}",
        flush=True,
    )
    t0 = time.perf_counter()
    result = run_agentic(
        train_df, test_df, target_col, positive_label, args.dataset, args.seed,
        dry_run=args.dry_run,
        optuna_trials=optuna_trials,
    )
    result.runtime_wallclock_sec = round(time.perf_counter() - t0, 3)

    print(f"\n{'=' * 70}\nRESULT: system=agentic dataset={args.dataset} seed={args.seed} optuna_trials={optuna_trials if optuna_trials is not None else 'adaptive'}")
    print(f"success={result.success} accuracy={result.accuracy} f1_weighted={result.f1_weighted} auc_roc={result.auc_roc}")
    print(f"iterations={result.iteration_count} used_fallback={result.used_fallback} "
          f"total_wallclock_sec={result.runtime_wallclock_sec}")
    print(f"stop_reason={result.extra_params.get('stop_reason')}")

    print(f"\n{'-' * 70}\nWALL-CLOCK BY STAGE (summed across all visits):")
    stage_wc = result.extra_params.get("stage_wallclock_sec", {})
    for stage, dt in stage_wc.items():
        print(f"  {stage:<20s} {dt:>8.3f}s")
    print(f"  {'TOTAL (nodes)':<20s} {sum(stage_wc.values()):>8.3f}s")
    optuna_calls = result.extra_params.get("optuna_wallclock_sec_per_call", [])
    print(f"\n  of which Optuna: {result.extra_params.get('optuna_wallclock_sec_total')}s "
          f"across {len(optuna_calls)} call(s) {optuna_calls}")
    print(f"  (Optuna time is counted inside model_selector's total above, not in addition to it)")

    print(f"\n{'-' * 70}\nTOKENS & COST BY AGENT:")
    for agent, usage in result.extra_params.get("token_usage_by_agent", {}).items():
        cost = result.extra_params["cost_by_agent_usd"].get(agent, 0.0)
        print(f"  {agent:<20s} in={usage.get('input_tokens', 0):>6d}  out={usage.get('output_tokens', 0):>6d}  ${cost:.6f}")
    print(f"  {'TOTAL':<20s} api_token_cost_usd=${result.api_token_cost_usd}")

    print(f"\n{'-' * 70}\nPER-ITERATION TRAIL (instruction -> what changed):")
    for entry in (result.iteration_metrics or []):
        acted_on = entry["acted_on_component"]
        print(f"\n  Iteration {entry['iteration']}:")
        if acted_on is None:
            print("    initial pass — no prior instruction, all agents decided fresh")
        else:
            print(f"    acted on previous round's diagnosis: weakest_component={acted_on!r}")
            print(f"      instruction consumed: {entry['acted_on_instruction']!r}")
        print(f"    tuning_case={entry.get('tuning_case')} model={entry['model']} best_params={entry['best_params']}")
        star = "  <-- reported as this run's result (Change 2: best iteration, not last)" if entry.get("selected_as_final") else ""
        print(f"    metrics: accuracy={entry.get('accuracy')} f1_weighted={entry.get('f1_weighted')} auc_roc={entry.get('auc_roc')} "
              f"recall_pos={entry.get('recall_pos')} precision_pos={entry.get('precision_pos')} threshold={entry.get('threshold')}{star}")
        print(f"    train_auc_roc={entry.get('train_auc_roc')} train_test_gap={entry.get('train_test_gap')} "
              f"overfitting={entry.get('overfitting_flag')} underfitting={entry.get('underfitting_flag')}")
        print(f"    use_class_weighting={entry.get('use_class_weighting')} class_weight_ratio={entry.get('class_weight_ratio')} "
              f"calibration_metric={entry.get('calibration_metric_used')}")
        d = entry.get("diagnosis", {})
        print(f"    this round's diagnosis -> weakest_component={d.get('weakest_component')!r}")
        print(f"      failure_mode: {d.get('failure_mode')}")
        print(f"      instruction for next round: {d.get('instruction')}")
        print(f"    stop_reason={entry.get('stop_reason')}")

    print(f"\n{'-' * 70}\nEVENTS:")
    for e in result.events:
        print(f"  [{e['stage']}] {e['type']}: {e['message'][:200]}")

    log_run(result, dry_run=args.dry_run)
