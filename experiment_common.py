"""Shared contract for all three systems: dataset registry, split, metrics, threshold calibration, result schema, MLflow logging."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split


DATA_DIR = Path(__file__).parent / "data"

# MLflow 3.x deprecates the filesystem backend; default to local SQLite unless a tracking URI is already set.
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).parent / 'mlflow.db'}")


def ensure_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    key = getpass.getpass("Enter your Anthropic API key (input hidden, not stored to disk): ").strip()
    if not key:
        raise SystemExit("No Anthropic API key provided. Set ANTHROPIC_API_KEY or re-run and paste one when prompted.")
    os.environ["ANTHROPIC_API_KEY"] = key


DATASETS: Dict[str, Dict[str, Any]] = {
    "telco_churn": {
        "path": DATA_DIR / "telco_churn.csv",
        "target": "Churn",
        "positive_label": "Yes",
        "drop_cols": ["customerID"],
    },
    "credit_default": {
        "path": DATA_DIR / "credit_default.csv",
        "target": "default.payment.next.month",
        "positive_label": 1,
        "drop_cols": [],
    },
    "stroke": {
        "path": DATA_DIR / "stroke.csv",
        "target": "stroke",
        "positive_label": 1,
        "drop_cols": ["id"],
    },
    "telco_churn_anon": {
        "path": DATA_DIR / "telco_churn_anon.csv",
        "target": "target",
        "positive_label": "Yes",
        "drop_cols": [],
    },
}

EXPERIMENT_DATASETS = ["telco_churn", "credit_default", "stroke"]

SIX_STAGES = [
    "profiling",
    "cleaning",
    "feature_engineering",
    "model_selection",
    "hyperparameter_tuning",
    "evaluation",
]

EXPERIMENT_NAME = "agentic-ds-framework"


def load_dataset(name: str) -> Tuple[pd.DataFrame, str]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS)}")
    spec = DATASETS[name]
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Place the raw CSV there before running experiments."
        )
    df = pd.read_csv(path)
    drop_cols = [c for c in spec["drop_cols"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    target = spec["target"]
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset '{name}' "
            f"(columns: {df.columns.tolist()})"
        )
    return df, target


# Called once per (dataset, seed); the same frames go to every system so splits are identical.
def make_split(
    df: pd.DataFrame, target_col: str, seed: int, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target_col]
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=y,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# Shared so every system binarizes the target the same way; independent encoders could flip the positive class.
def encode_target(y_train: pd.Series, y_test: pd.Series, positive_label: Any):
    y_train_enc = (y_train == positive_label).astype(int).to_numpy()
    y_test_enc = (y_test == positive_label).astype(int).to_numpy()
    return y_train_enc, y_test_enc


def compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        # Positive-class metrics; weighted averages hide minority-class failures.
        "recall_pos": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision_pos": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
    }
    auc = None
    if y_proba is not None:
        try:
            proba = np.asarray(y_proba)
            if proba.ndim == 2 and proba.shape[1] == 2:
                auc = roc_auc_score(y_true, proba[:, 1])
            elif proba.ndim == 1:
                auc = roc_auc_score(y_true, proba)
        except ValueError:
            auc = None
    metrics["auc_roc"] = round(float(auc), 4) if auc is not None else None
    return metrics


THRESHOLD_CALIBRATION_FOLDS = 5
# 'f1' mirrors H2O's default; 'f2' weights recall higher.
THRESHOLD_CALIBRATION_METRIC = "f1"
THRESHOLD_CALIBRATION_GRID_POINTS = 1001


# Takes no test data; callers must pass only out-of-fold training predictions.
def find_best_threshold(y_true, y_proba, metric: str = THRESHOLD_CALIBRATION_METRIC) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    beta = 2.0 if metric == "f2" else 1.0
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, THRESHOLD_CALIBRATION_GRID_POINTS):
        preds = (y_proba >= t).astype(int)
        score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


# Training data only. `estimator` must be unfitted so cross_val_predict yields genuinely out-of-fold probabilities.
def calibrate_threshold_oof(
    estimator, X_train, y_train, seed: int, metric: str = THRESHOLD_CALIBRATION_METRIC,
    subsample_row_threshold: Optional[int] = None, subsample_size: Optional[int] = None,
) -> float:
    X_fit, y_fit = X_train, y_train
    if subsample_row_threshold and subsample_size and len(X_train) > subsample_row_threshold:
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train, train_size=subsample_size, stratify=y_train, random_state=seed,
        )
    cv = StratifiedKFold(n_splits=THRESHOLD_CALIBRATION_FOLDS, shuffle=True, random_state=seed)
    proba_oof = cross_val_predict(estimator, X_fit, y_fit, cv=cv, method="predict_proba")[:, 1]
    return find_best_threshold(y_fit, proba_oof, metric=metric)


def automation_score(stage_flags: Dict[str, bool]) -> float:
    if not stage_flags:
        return 0.0
    return round(sum(1 for v in stage_flags.values() if v) / len(stage_flags), 4)


@dataclass
class RunResult:
    system: str
    dataset: str
    seed: int
    success: bool
    accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None
    auc_roc: Optional[float] = None
    recall_pos: Optional[float] = None
    precision_pos: Optional[float] = None
    threshold_used: Optional[float] = None
    runtime_wallclock_sec: Optional[float] = None
    model_train_time_sec: Optional[float] = None
    used_fallback: bool = False
    failure_cause: Optional[str] = None
    stage_automation: Dict[str, bool] = field(default_factory=lambda: {s: False for s in SIX_STAGES})
    events: List[Dict[str, str]] = field(default_factory=list)
    iteration_count: Optional[int] = None
    iteration_metrics: Optional[List[Dict[str, Any]]] = None
    api_token_cost_usd: Optional[float] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


def log_run(result: RunResult, dry_run: bool = False) -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"{result.system}_{result.dataset}_seed{result.seed}"
    with mlflow.start_run(run_name=run_name):
        tags = {
            "dataset": result.dataset,
            "seed": str(result.seed),
            "system": result.system,
            "dry_run": str(dry_run),
        }
        if result.failure_cause:
            tags["failure_cause"] = result.failure_cause[:250]
        if result.used_fallback:
            tags["used_fallback"] = "true"
        mlflow.set_tags(tags)

        if result.extra_params:
            mlflow.log_params({k: str(v) for k, v in result.extra_params.items()})

        metrics: Dict[str, float] = {
            "success": int(result.success),
            "used_fallback": int(result.used_fallback),
            "automation_score": automation_score(result.stage_automation),
        }
        for key in ("accuracy", "f1_weighted", "auc_roc", "recall_pos", "precision_pos", "threshold_used",
                    "runtime_wallclock_sec", "model_train_time_sec"):
            value = getattr(result, key)
            if value is not None:
                metrics[key] = value
        for stage, done in result.stage_automation.items():
            metrics[f"automation_stage_{stage}"] = int(done)
        if result.iteration_count is not None:
            metrics["iteration_count"] = result.iteration_count
        if result.api_token_cost_usd is not None:
            metrics["api_token_cost_usd"] = result.api_token_cost_usd
        mlflow.log_metrics(metrics)

        if result.iteration_metrics:
            for i, m in enumerate(result.iteration_metrics):
                for k, v in m.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        mlflow.log_metric(f"iter_{k}", v, step=i)
            mlflow.log_dict({"iterations": result.iteration_metrics}, "iteration_metrics.json")

        if result.events:
            mlflow.log_dict({"events": result.events}, "events.json")
