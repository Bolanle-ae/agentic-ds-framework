import argparse
import time
import traceback

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from experiment_common import (
    DATASETS,
    RunResult,
    SIX_STAGES,
    compute_metrics,
    encode_target,
    load_dataset,
    log_run,
    make_split,
)


def _fallback_model(train_df, test_df, target_col, positive_label, seed, result: RunResult, reason: str):
    result.used_fallback = True
    result.events.append({"stage": "model_selection", "type": "fallback", "message": reason})
    for stage in ("cleaning", "feature_engineering", "model_selection", "hyperparameter_tuning"):
        result.stage_automation[stage] = False

    X_train = train_df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
    X_test = (
        test_df.drop(columns=[target_col])
        .select_dtypes(include=[np.number])
        .reindex(columns=X_train.columns, fill_value=0)
        .fillna(0)
    )
    y_train, y_test = encode_target(train_df[target_col], test_df[target_col], positive_label)

    model = RandomForestClassifier(random_state=seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    result.stage_automation["evaluation"] = True
    result.accuracy = metrics["accuracy"]
    result.f1_weighted = metrics["f1_weighted"]
    result.auc_roc = metrics["auc_roc"]
    result.model_train_time_sec = round(train_time, 4)
    result.success = True


def run_h2o(
    train_df, test_df, target_col, positive_label, dataset_name, seed,
    max_runtime_secs: int = 600, max_models: int = 20,
) -> RunResult:
    result = RunResult(
        system="h2o", dataset=dataset_name, seed=seed, success=False,
        stage_automation={s: False for s in SIX_STAGES},
        extra_params={"max_runtime_secs": max_runtime_secs, "max_models": max_models},
    )

    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError as e:
        _fallback_model(
            train_df, test_df, target_col, positive_label, seed, result,
            reason=f"h2o package not importable: {e}",
        )
        return result

    try:
        h2o.init(nthreads=-1, max_mem_size="2G")
    except Exception as e:
        result.failure_cause = f"h2o.init() failed: {type(e).__name__}: {e}"
        _fallback_model(
            train_df, test_df, target_col, positive_label, seed, result,
            reason=f"h2o.init() failed (commonly: no Java runtime on this host): {e}",
        )
        return result

    try:
        y_train, y_test = encode_target(train_df[target_col], test_df[target_col], positive_label)

        train_h2o_df = train_df.drop(columns=[target_col]).copy()
        train_h2o_df["__target__"] = y_train
        test_h2o_df = test_df.drop(columns=[target_col]).copy()
        test_h2o_df["__target__"] = y_test

        train_h2o = h2o.H2OFrame(train_h2o_df)
        test_h2o = h2o.H2OFrame(test_h2o_df)
        train_h2o["__target__"] = train_h2o["__target__"].asfactor()
        test_h2o["__target__"] = test_h2o["__target__"].asfactor()

        x = [c for c in train_h2o.columns if c != "__target__"]

        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            seed=seed,
            sort_metric="AUC",
        )
        t0 = time.perf_counter()
        aml.train(x=x, y="__target__", training_frame=train_h2o)
        train_time = time.perf_counter() - t0

        for stage in ("cleaning", "feature_engineering", "model_selection", "hyperparameter_tuning"):
            result.stage_automation[stage] = True

        preds = aml.leader.predict(test_h2o).as_data_frame()
        y_pred = preds["predict"].to_numpy().astype(int)
        proba_col = "p1" if "p1" in preds.columns else preds.columns[-1]
        y_proba = preds[proba_col].to_numpy()

        metrics = compute_metrics(y_test, y_pred, y_proba)
        result.stage_automation["evaluation"] = True
        result.accuracy = metrics["accuracy"]
        result.f1_weighted = metrics["f1_weighted"]
        result.auc_roc = metrics["auc_roc"]
        result.recall_pos = metrics["recall_pos"]
        result.precision_pos = metrics["precision_pos"]
        try:
            # H2O's predict() already applies its own F1-optimal threshold; this only records it.
            result.threshold_used = float(aml.leader.find_threshold_by_max_metric("f1"))
        except Exception:
            result.threshold_used = None
        result.model_train_time_sec = round(train_time, 4)
        result.extra_params["leader_model"] = aml.leader.model_id
        result.success = True

    except Exception as e:
        result.events.append({"stage": "h2o_automl", "type": "failure", "message": traceback.format_exc()})
        _fallback_model(
            train_df, test_df, target_col, positive_label, seed, result,
            reason=f"H2O AutoML run failed: {type(e).__name__}: {e}",
        )
    finally:
        try:
            h2o.cluster().shutdown()
        except Exception:
            pass

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the H2O AutoML baseline standalone.")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df, target_col = load_dataset(args.dataset)
    positive_label = DATASETS[args.dataset]["positive_label"]
    train_df, test_df = make_split(df, target_col, args.seed)

    t0 = time.perf_counter()
    result = run_h2o(
        train_df, test_df, target_col, positive_label, args.dataset, args.seed,
        max_runtime_secs=60 if args.dry_run else 600,
        max_models=20,
    )
    result.runtime_wallclock_sec = round(time.perf_counter() - t0, 3)

    print(result)
    log_run(result, dry_run=args.dry_run)
