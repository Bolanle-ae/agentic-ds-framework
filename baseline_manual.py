import argparse
import time
import traceback

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experiment_common import (
    DATASETS,
    RunResult,
    SIX_STAGES,
    THRESHOLD_CALIBRATION_METRIC,
    calibrate_threshold_oof,
    compute_metrics,
    encode_target,
    load_dataset,
    log_run,
    make_split,
)


def run_manual(train_df, test_df, target_col, positive_label, dataset_name, seed) -> RunResult:
    result = RunResult(
        system="manual", dataset=dataset_name, seed=seed, success=False,
        stage_automation={s: False for s in SIX_STAGES},
    )

    try:
        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        y_train, y_test = encode_target(train_df[target_col], test_df[target_col], positive_label)

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ])
        model = RandomForestClassifier(random_state=seed)
        clf = Pipeline([("prep", preprocessor), ("model", model)])

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # Unfitted clone of the whole pipeline, so each fold's preprocessing statistics come from that fold's training rows only.
        calibration_clf = clone(clf)
        threshold = calibrate_threshold_oof(
            calibration_clf, X_train, y_train, seed, metric=THRESHOLD_CALIBRATION_METRIC,
        )

        y_proba = clf.predict_proba(X_test)
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        result.accuracy = metrics["accuracy"]
        result.f1_weighted = metrics["f1_weighted"]
        result.auc_roc = metrics["auc_roc"]
        result.recall_pos = metrics["recall_pos"]
        result.precision_pos = metrics["precision_pos"]
        result.threshold_used = threshold
        result.model_train_time_sec = round(train_time, 4)
        result.stage_automation["evaluation"] = True
        result.success = True

    except Exception as e:
        result.failure_cause = f"{type(e).__name__}: {e}"
        result.events.append({"stage": "unknown", "type": "failure", "message": traceback.format_exc()})
        result.success = False

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the manual sklearn baseline standalone.")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, target_col = load_dataset(args.dataset)
    positive_label = DATASETS[args.dataset]["positive_label"]
    train_df, test_df = make_split(df, target_col, args.seed)

    t0 = time.perf_counter()
    result = run_manual(train_df, test_df, target_col, positive_label, args.dataset, args.seed)
    result.runtime_wallclock_sec = round(time.perf_counter() - t0, 3)

    print(result)
    log_run(result)
