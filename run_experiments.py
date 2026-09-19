import argparse
import itertools
import time
import traceback

import baseline_h2o
import baseline_manual
import pipeline as agentic_pipeline
from experiment_common import (
    DATASETS,
    EXPERIMENT_DATASETS,
    EXPERIMENT_NAME,
    RunResult,
    automation_score,
    ensure_api_key,
    load_dataset,
    log_run,
    make_split,
)

SYSTEMS = ["agentic", "manual", "h2o"]
SEEDS = [42, 123, 2024]


def run_one(system: str, train_df, test_df, target_col, positive_label, dataset_name, seed, dry_run: bool) -> RunResult:
    t0 = time.perf_counter()
    try:
        if system == "agentic":
            result = agentic_pipeline.run_agentic(
                train_df, test_df, target_col, positive_label, dataset_name, seed,
                dry_run=dry_run,
                # None lets model_selector use the adaptive first/retune trial counts.
                optuna_trials=2 if dry_run else None,
            )
        elif system == "manual":
            result = baseline_manual.run_manual(train_df, test_df, target_col, positive_label, dataset_name, seed)
        elif system == "h2o":
            result = baseline_h2o.run_h2o(
                train_df, test_df, target_col, positive_label, dataset_name, seed,
                max_runtime_secs=60 if dry_run else 600,
                max_models=20,
            )
        else:
            raise ValueError(f"Unknown system: {system}")
    except Exception as e:
        result = RunResult(system=system, dataset=dataset_name, seed=seed, success=False,
                            failure_cause=f"{type(e).__name__}: {e}")
        result.events.append({"stage": "orchestration", "type": "failure", "message": traceback.format_exc()})

    result.runtime_wallclock_sec = round(time.perf_counter() - t0, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the agentic-ds-framework experiment grid.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Tiny budgets (2 Optuna trials, 1 iteration cap, 60s H2O budget) to verify the harness end to end.")
    parser.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    parser.add_argument("--datasets", nargs="+", default=EXPERIMENT_DATASETS, choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    if "agentic" in args.systems:
        ensure_api_key()

    total = len(args.datasets) * len(args.seeds) * len(args.systems)
    print(f"agentic-ds-framework: {total} run(s) across {len(args.datasets)} dataset(s) x "
          f"{len(args.seeds)} seed(s) x {len(args.systems)} system(s)"
          f"{' [DRY RUN — tiny budgets]' if args.dry_run else ''}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}\n")

    results = []
    run_idx = 0
    for dataset_name in args.datasets:
        df, target_col = load_dataset(dataset_name)
        positive_label = DATASETS[dataset_name]["positive_label"]

        for seed in args.seeds:
            # Generated once per (dataset, seed) and shared by all three systems.
            train_df, test_df = make_split(df, target_col, seed)
            print(f"=== {dataset_name} | seed={seed} | train={len(train_df)} test={len(test_df)} ===")

            for system in args.systems:
                run_idx += 1
                print(f"  [{run_idx}/{total}] {system}...", end=" ", flush=True)
                result = run_one(system, train_df, test_df, target_col, positive_label, dataset_name, seed, args.dry_run)
                log_run(result, dry_run=args.dry_run)
                results.append(result)

                status = "OK" if result.success else f"FAILED ({result.failure_cause})"
                print(
                    f"{status} | acc={result.accuracy} f1={result.f1_weighted} auc={result.auc_roc} "
                    f"recall={result.recall_pos} precision={result.precision_pos} threshold={result.threshold_used} "
                    f"| automation={automation_score(result.stage_automation)} | fallback={result.used_fallback} "
                    f"| wallclock={result.runtime_wallclock_sec}s"
                )

    n_ok = sum(r.success for r in results)
    n_fallback = sum(r.used_fallback for r in results)
    print(f"\n{'=' * 70}")
    print(f"Done: {n_ok}/{len(results)} succeeded, {n_fallback} used a fallback path.")
    print(f"Logged to MLflow experiment '{EXPERIMENT_NAME}'. Run `mlflow ui` to inspect.")

    failed = [r for r in results if not r.success]
    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - {r.system}/{r.dataset}/seed{r.seed}: {r.failure_cause}")


if __name__ == "__main__":
    main()
