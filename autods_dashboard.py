import io
import os
import pickle
import time

import pandas as pd
import streamlit as st

from experiment_common import DATASETS, automation_score, load_dataset, make_split
from pipeline import (
    MAX_ITERATIONS_FULL,
    NO_IMPROVEMENT_EPSILON,
    NO_IMPROVEMENT_PATIENCE,
    PERFORMANCE_TARGET,
    build_result_from_state,
    run_agentic_streaming,
)

st.set_page_config(page_title="AutoDS", page_icon="🔶", layout="wide")

AGENT_ORDER = ["profiler", "cleaner", "feature_engineer", "model_selector", "evaluator", "reflection"]
AGENT_LABELS = {
    "profiler": "1. Data Profiling",
    "cleaner": "2. Data Cleaning",
    "feature_engineer": "3. Feature Engineering",
    "model_selector": "4. Model Selection",
    "evaluator": "5. Evaluation Agent",
    "reflection": "6. Reflection Agent",
}


for key, default in {
    "page": "Setup", "pending_run": None, "last_state": None,
    "run_result": None, "run_config": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def compute_agent_statuses(state: dict) -> dict:
    stopped = state.get("stop_reason") is not None
    current_iter = state["iteration"] if stopped else state["iteration"] + 1

    counts, last_duration = {}, {}
    for entry in state["stage_timings"]:
        counts[entry["stage"]] = counts.get(entry["stage"], 0) + 1
        last_duration[entry["stage"]] = entry["duration_sec"]

    statuses, found_running = {}, False
    for agent in AGENT_ORDER:
        expected = 1 if agent == "profiler" else max(current_iter, 1)
        if counts.get(agent, 0) >= expected:
            statuses[agent] = ("completed", last_duration.get(agent))
        elif not found_running:
            statuses[agent] = ("running", None)
            found_running = True
        else:
            statuses[agent] = ("pending", None)
    return statuses


def chart_points(state: dict) -> pd.DataFrame:
    rows = [
        {"iteration": e["iteration"], "AUC-ROC": e.get("auc_roc"), "F1 Score": e.get("f1_weighted")}
        for e in state["iteration_history"]
    ]
    metrics = state.get("metrics")
    current_iter = state["iteration"] + 1
    if metrics and (not rows or rows[-1]["iteration"] != current_iter):
        rows.append({"iteration": current_iter, "AUC-ROC": metrics.get("auc_roc"), "F1 Score": metrics.get("f1_weighted")})
    return pd.DataFrame(rows)


def render_agent_pipeline(state: dict):
    statuses = compute_agent_statuses(state)
    cols = st.columns(6)
    icon = {"completed": "✅", "running": "🟠", "pending": "⚪"}
    for col, agent in zip(cols, AGENT_ORDER):
        status, duration = statuses[agent]
        with col:
            st.markdown(f"**{icon[status]} {AGENT_LABELS[agent]}**")
            if status == "completed":
                st.caption(f"Completed · {duration:.2f}s" if duration is not None else "Completed")
            elif status == "running":
                st.caption("Running…")
            else:
                st.caption("Pending")


def render_metrics_row(state: dict):
    metrics = state.get("metrics") or {}
    history = state["iteration_history"]
    prev = history[-1] if history else None

    c1, c2, c3, c4 = st.columns(4)
    auc = metrics.get("auc_roc")
    f1 = metrics.get("f1_weighted")
    c1.metric("AUC-ROC", f"{auc:.3f}" if auc is not None else "—",
               delta=(f"{auc - prev.get('auc_roc', auc):+.3f}" if auc is not None and prev and prev.get("auc_roc") is not None else None))
    c2.metric("F1 Score", f"{f1:.3f}" if f1 is not None else "—",
               delta=(f"{f1 - prev.get('f1_weighted', f1):+.3f}" if f1 is not None and prev and prev.get("f1_weighted") is not None else None))
    tokens = state.get("token_usage", {})
    total_cost = sum(
        u.get("input_tokens", 0) / 1e6 * 1.00 + u.get("output_tokens", 0) / 1e6 * 5.00
        for u in tokens.values()
    )
    c3.metric("API cost so far", f"${total_cost:.4f}")
    c4.metric("Iteration", f"{state['iteration'] + (0 if state.get('stop_reason') else 1)} / {state['max_iterations']}")


def render_reflection_panel(state: dict):
    st.subheader(f"Reflection · after iteration {state['iteration']}")
    if state.get("weakest_component") is None:
        st.info("No diagnosis yet — reflection hasn't completed an iteration.")
        return
    st.markdown(f"**Weakest component:** `{state['weakest_component']}`")
    st.markdown("**Diagnosis**")
    st.write(state.get("current_failure_mode") or "—")
    st.markdown("**Instruction for next attempt**")
    st.write(state.get("current_instruction") or "—")
    if state.get("stop_reason"):
        st.caption(f"Loop stopped: `{state['stop_reason']}` — this instruction was not consumed.")


with st.sidebar:
    st.markdown("## 🔶 AutoDS")
    page = st.radio("Workspace", ["Setup", "Live Run", "Reports"],
                     index=["Setup", "Live Run", "Reports"].index(st.session_state.page), label_visibility="collapsed")
    st.session_state.page = page
    st.divider()
    api_key_input = st.text_input(
        "Anthropic API Key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Used only for this session; never written to disk.",
    )


if st.session_state.page == "Setup":
    st.title("Setup")
    st.caption("Start a new automated modeling run")

    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    spec = DATASETS[dataset_name]
    c1, c2 = st.columns(2)
    c1.text_input("Target column", value=spec["target"], disabled=True)
    seed = c2.number_input("Random seed", value=42, step=1)

    st.markdown("#### Loop settings")
    c1, c2 = st.columns(2)
    max_iterations = c1.slider("Max iterations", 1, 5, MAX_ITERATIONS_FULL)
    target_auc = c2.slider("Target AUC-ROC", 0.50, 0.99, PERFORMANCE_TARGET, 0.01)
    optuna_trials = st.slider(
        "Optuna trials", 2, 25, 10,
        help="25 is the real full-budget value used in the actual experiment. Lower = faster demo.",
    )
    # Rough seconds per Optuna trial, from measured runs.
    est_low = 0.02 * optuna_trials
    est_high = 0.7 * optuna_trials
    st.caption(
        f"⏱ Estimated: {est_low * 1:.0f}s–{est_high * max_iterations:.0f}s "
        f"(dominated by Optuna; wildly dataset-dependent — see conversation history for why)."
    )

    if st.button("Run Pipeline", type="primary", disabled=not api_key_input, use_container_width=True):
        st.session_state.pending_run = {
            "dataset_name": dataset_name, "seed": int(seed),
            "max_iterations": max_iterations, "target_auc": target_auc,
            "optuna_trials": optuna_trials, "api_key": api_key_input,
        }
        st.session_state.page = "Live Run"
        st.rerun()
    if not api_key_input:
        st.warning("Enter an Anthropic API key to enable the run button.")


elif st.session_state.page == "Live Run":
    st.title("Live Run")

    if st.session_state.pending_run is not None:
        cfg = st.session_state.pending_run
        st.session_state.pending_run = None
        os.environ["ANTHROPIC_API_KEY"] = cfg["api_key"]

        st.caption(f"{cfg['dataset_name']} · seed {cfg['seed']}")
        pipeline_box = st.empty()
        metrics_box = st.empty()
        chart_box = st.empty()
        reflection_box = st.empty()

        df, target_col = load_dataset(cfg["dataset_name"])
        positive_label = DATASETS[cfg["dataset_name"]]["positive_label"]
        train_df, test_df = make_split(df, target_col, cfg["seed"])

        t0 = time.perf_counter()
        last_state = None
        for state in run_agentic_streaming(
            train_df, test_df, target_col, positive_label, cfg["dataset_name"], cfg["seed"],
            max_iterations=cfg["max_iterations"], optuna_trials=cfg["optuna_trials"],
            performance_target=cfg["target_auc"], no_improvement_epsilon=NO_IMPROVEMENT_EPSILON,
        ):
            last_state = state
            with pipeline_box.container():
                render_agent_pipeline(state)
            with metrics_box.container():
                render_metrics_row(state)
            points = chart_points(state)
            if not points.empty:
                chart_box.line_chart(points.set_index("iteration")[["AUC-ROC", "F1 Score"]])
            with reflection_box.container():
                render_reflection_panel(state)

        wallclock = round(time.perf_counter() - t0, 3)
        result = build_result_from_state(
            last_state, cfg["dataset_name"], cfg["seed"], cfg["max_iterations"],
            cfg["optuna_trials"], cfg["target_auc"],
        )
        result.runtime_wallclock_sec = wallclock

        st.session_state.last_state = last_state
        st.session_state.run_result = result
        st.session_state.run_config = cfg
        st.success(f"Run complete — stop_reason: `{result.extra_params.get('stop_reason')}`")
        if st.button("View Reports →", type="primary"):
            st.session_state.page = "Reports"
            st.rerun()

    elif st.session_state.last_state is not None:
        state = st.session_state.last_state
        st.caption(f"{st.session_state.run_config['dataset_name']} · seed {st.session_state.run_config['seed']} (last completed run)")
        render_agent_pipeline(state)
        render_metrics_row(state)
        points = chart_points(state)
        if not points.empty:
            st.line_chart(points.set_index("iteration")[["AUC-ROC", "F1 Score"]])
        render_reflection_panel(state)
    else:
        st.info("Start a run from **Setup**.")


elif st.session_state.page == "Reports":
    st.title("Reports")
    result = st.session_state.run_result
    state = st.session_state.last_state

    if result is None:
        st.info("No completed run yet — start one from **Setup**.")
    else:
        cfg = st.session_state.run_config
        st.caption(f"{cfg['dataset_name']} · seed {cfg['seed']}")

        points = chart_points(state)
        best_auc = points["AUC-ROC"].max() if not points.empty else None
        best_auc_iter = int(points.loc[points["AUC-ROC"].idxmax(), "iteration"]) if not points.empty else None
        best_f1 = points["F1 Score"].max() if not points.empty else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best AUC-ROC", f"{best_auc:.3f}" if best_auc is not None else "—",
                   delta=f"iter {best_auc_iter}" if best_auc_iter else None)
        c2.metric("Best F1 Score", f"{best_f1:.3f}" if best_f1 is not None else "—")
        c3.metric("Total API cost", f"${result.api_token_cost_usd:.4f}" if result.api_token_cost_usd else "$0.0000")
        mins, secs = divmod(int(result.runtime_wallclock_sec or 0), 60)
        c4.metric("Total runtime", f"{mins:02d}:{secs:02d}")

        st.markdown("#### Metric by iteration")
        if not points.empty:
            st.line_chart(points.set_index("iteration")[["AUC-ROC", "F1 Score"]])

        st.markdown("#### Automation score")
        st.progress(automation_score(result.stage_automation), text=f"{automation_score(result.stage_automation):.0%} of stages completed without human intervention")
        st.json(result.stage_automation, expanded=False)

        st.markdown("#### Per-iteration trail")
        for entry in result.iteration_metrics or []:
            with st.expander(f"Iteration {entry['iteration']} — {entry.get('model')}"):
                st.write(f"Metrics: accuracy={entry.get('accuracy')}, f1={entry.get('f1_weighted')}, auc={entry.get('auc_roc')}")
                d = entry.get("diagnosis", {})
                if d:
                    st.write(f"**Diagnosis:** {d.get('failure_mode')}")
                    st.write(f"**Instruction:** {d.get('instruction')}")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            import dataclasses
            import json as _json
            payload = _json.dumps(dataclasses.asdict(result), indent=2, default=str)
            st.download_button("⬇ Export results as JSON", data=payload,
                                file_name=f"autods_{cfg['dataset_name']}_seed{cfg['seed']}.json",
                                mime="application/json", use_container_width=True)
        with c2:
            model = state.get("model") if state else None
            if model is not None:
                buf = io.BytesIO()
                pickle.dump(model, buf)
                st.download_button("⬇ Download trained model (.pkl)", data=buf.getvalue(),
                                    file_name=f"autods_{cfg['dataset_name']}_seed{cfg['seed']}_model.pkl",
                                    mime="application/octet-stream", use_container_width=True)
            else:
                st.button("⬇ Download trained model (.pkl)", disabled=True, use_container_width=True)
