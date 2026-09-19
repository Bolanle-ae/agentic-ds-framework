"""Exploratory probe: runs the agentic pipeline once per dataset at seed 42 with the reflection model set by REFLECTION_LLM_MODEL, logging to a separate MLflow experiment."""

import argparse
import json
import time

import experiment_common
import pipeline as agentic_pipeline
from experiment_common import DATASETS, EXPERIMENT_DATASETS, load_dataset, log_run, make_split

PROBE_EXPERIMENT_NAME = "reflection-sonnet-test"
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="label for this batch, e.g. 'haiku' or 'sonnet'")
    args = parser.parse_args()

    # Process-local override, so the main experiment_common.py and its experiment are untouched.
    experiment_common.EXPERIMENT_NAME = PROBE_EXPERIMENT_NAME

    reflection_model = agentic_pipeline.REFLECTION_LLM_MODEL
    print(f"probe batch: tag={args.tag!r} reflection_model={reflection_model!r} "
          f"other_agents_model={agentic_pipeline.LLM_MODEL!r} seed={SEED}")
    print(f"logging to ISOLATED MLflow experiment: {PROBE_EXPERIMENT_NAME}\n")

    for dataset_name in EXPERIMENT_DATASETS:
        df, target_col = load_dataset(dataset_name)
        positive_label = DATASETS[dataset_name]["positive_label"]
        train_df, test_df = make_split(df, target_col, SEED)

        print(f"=== {dataset_name} | seed={SEED} | reflection={args.tag} ===", flush=True)
        t0 = time.perf_counter()
        result = agentic_pipeline.run_agentic(
            train_df, test_df, target_col, positive_label, dataset_name, SEED, dry_run=False,
        )
        result.runtime_wallclock_sec = round(time.perf_counter() - t0, 3)
        result.extra_params["reflection_model"] = reflection_model
        result.extra_params["probe_tag"] = args.tag

        log_run(result, dry_run=False)

        out_path = f"probe_{args.tag}_{dataset_name}_seed{SEED}.json"
        with open(out_path, "w") as f:
            json.dump({
                "dataset": dataset_name, "seed": SEED, "tag": args.tag, "reflection_model": reflection_model,
                "success": result.success, "auc_roc": result.auc_roc, "recall_pos": result.recall_pos,
                "precision_pos": result.precision_pos, "f1_weighted": result.f1_weighted,
                "accuracy": result.accuracy, "used_fallback": result.used_fallback,
                "wallclock": result.runtime_wallclock_sec, "extra_params": result.extra_params,
                "iteration_metrics": result.iteration_metrics, "events": result.events,
            }, f, indent=2, default=str)

        print(f"  -> {out_path}")
        print(f"  OK | auc={result.auc_roc} recall={result.recall_pos} precision={result.precision_pos} "
              f"f1={result.f1_weighted} acc={result.accuracy} | wallclock={result.runtime_wallclock_sec}s "
              f"| logged_cost=${result.api_token_cost_usd} (Haiku-rate for ALL agents, see report for corrected Sonnet cost)\n")


if __name__ == "__main__":
    main()
