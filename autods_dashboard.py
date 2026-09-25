import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import autods_core as core

APP_DIR = Path(__file__).parent
MAX_CONCURRENT_LIVE_RUNS = 2
MAX_RUN_SECONDS = 30 * 60
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_TARGET_AUC = 0.85

st.set_page_config(page_title="AutoDS", page_icon="🔶", layout="wide", initial_sidebar_state="expanded")

LOGO = (
    '<svg viewBox="0 0 52 48" aria-hidden="true"><defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#C9CACC"/><stop offset="1" stop-color="#7A7C80"/></linearGradient></defs>'
    '<path d="M26 2l23 17-9 26H12L3 19z" fill="#FF8A3D"/>'
    '<path d="M26 11l15 11-5.5 17h-19L11 22z" fill="url(#g)"/>'
    '<path d="M19 45l7-19 7 19z" fill="#FF8A3D"/></svg>'
)
FILE_ICON = (
    '<svg viewBox="0 0 56 66" aria-hidden="true"><path d="M4 4h32l16 16v42H4z" fill="#FF8A3D"/>'
    '<path d="M36 4v16h16z" fill="#FFC79D"/><path d="M14 34h28M14 43h28M14 52h28" stroke="#fff" stroke-width="4" stroke-linecap="round"/></svg>'
)
CHECK_ICON = (
    '<svg class="ok" viewBox="0 0 68 68" aria-hidden="true"><circle cx="34" cy="34" r="30" fill="none" stroke="#7DBB8E" stroke-width="3" stroke-dasharray="7 6"/>'
    '<path d="M20 35l10 10 19-20" fill="none" stroke="#3B9A4A" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
CLOCK_ICON = (
    '<svg viewBox="0 0 26 26" aria-hidden="true"><circle cx="13" cy="13" r="10.5" fill="none" stroke="#FF8A3D" stroke-width="2.5"/>'
    '<path d="M13 7v6.5l4 2.5" fill="none" stroke="#FF8A3D" stroke-width="2.5" stroke-linecap="round"/></svg>'
)

for key, default in {
    "page": "Setup", "run": None, "report": None, "seed": 42, "dataset": None,
    "max_iterations": DEFAULT_MAX_ITERATIONS, "target_auc": DEFAULT_TARGET_AUC,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource
def process_registry() -> dict:
    return {}


@st.cache_data(show_spinner=False)
def inspect_csv(data: bytes) -> dict:
    df = pd.read_csv(io.BytesIO(data))
    return {"columns": list(df.columns), "rows": len(df)}


def live_runs_active() -> int:
    return sum(1 for p in process_registry().values() if p["mode"] == "live" and p["proc"].poll() is None)


def go(page: str):
    st.session_state.page = page


def kill_run(run: dict) -> None:
    entry = process_registry().get(run["id"])
    if entry and entry["proc"].poll() is None:
        entry["proc"].kill()


def start_run(mode: str, meta: dict, api_key: str = "", data: bytes = b"") -> None:
    previous = st.session_state.run
    if previous and not previous.get("finished"):
        kill_run(previous)
    run_dir = Path(tempfile.mkdtemp(prefix="autods_"))
    (run_dir / "events.jsonl").touch()
    env = dict(os.environ)
    if mode == "live":
        (run_dir / "data.csv").write_bytes(data)
        (run_dir / "config.json").write_text(json.dumps({
            "file_name": meta["file_name"], "target_column": meta["target_column"],
            "positive_label": meta["positive_label"], "seed": meta["seed"],
            "max_iterations": meta["max_iterations"], "performance_target": meta["performance_target"],
        }))
        env["ANTHROPIC_API_KEY"] = api_key
    else:
        env.pop("ANTHROPIC_API_KEY", None)
    log = open(run_dir / "worker.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(APP_DIR / "autods_worker.py"), mode, str(run_dir)],
        cwd=APP_DIR, env=env, stdout=log, stderr=log,
    )
    run_id = run_dir.name
    process_registry()[run_id] = {"proc": proc, "mode": mode}
    st.session_state.run = {
        "id": run_id, "dir": str(run_dir), "mode": mode, "meta": meta,
        "started": time.time(), "finished": False, "stopped": False, "result": None, "error": None,
    }
    st.session_state.report = None
    st.session_state.page = "Live Run"


def read_events(run: dict) -> dict:
    view = {"state": None, "result": None, "error": None, "last_ts": run["started"]}
    path = Path(run["dir"]) / "events.jsonl"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return view
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        view["last_ts"] = event.get("ts", view["last_ts"])
        if event["type"] == "state":
            view["state"] = event["state"]
        elif event["type"] == "done":
            view["result"] = event["result"]
        elif event["type"] == "error":
            view["error"] = event["message"]
    return view


def initial_state(max_iterations: int) -> dict:
    return {
        "iteration": 0, "max_iterations": max_iterations, "stage_timings": [], "iteration_history": [],
        "metrics": None, "stop_reason": None, "weakest_component": None,
        "current_failure_mode": None, "current_instruction": None,
    }


def refresh_run(run: dict) -> dict:
    view = read_events(run)
    entry = process_registry().get(run["id"])
    alive = bool(entry) and entry["proc"].poll() is None
    if alive and time.time() - run["started"] > MAX_RUN_SECONDS:
        kill_run(run)
        run["error"] = f"Run stopped after {MAX_RUN_SECONDS // 60} minutes."
        alive = False
    if view["result"] is not None:
        run["finished"], run["result"] = True, view["result"]
        st.session_state.report = {"result": view["result"], "meta": run["meta"], "run_id": run["id"]}
    elif view["error"]:
        run["finished"], run["error"] = True, view["error"]
    elif not alive and not run["finished"]:
        run["finished"] = True
        if not run["stopped"] and not run["error"]:
            tail = ""
            try:
                tail = (Path(run["dir"]) / "worker.log").read_text()[-400:]
            except OSError:
                pass
            run["error"] = "The run ended unexpectedly." + (f" {tail.strip()}" if tail.strip() else "")
    return view


def chrome(live: bool = False):
    st.markdown(f"<style>{(APP_DIR / 'assets' / 'autods.css').read_text()}</style>", unsafe_allow_html=True)
    if live:
        st.markdown(
            "<style>section[data-testid='stSidebar']{display:none !important}"
            ".block-container{padding-left:44px !important}</style>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="topbar"><div class="brand">{LOGO}<span>AutoDS</span></div>'
        + ('' if live else '<div class="avatar" title="AutoDS">BA</div>')
        + '</div>',
        unsafe_allow_html=True,
    )


def set_dataset(name: str, data: bytes, target: str, positive=None) -> None:
    info = inspect_csv(data)
    st.session_state.dataset = {"name": name, "data": data, "target": target, "positive": positive}
    st.session_state.target_col = target if target in info["columns"] else info["columns"][-1]


def choose_sample(spec: dict) -> None:
    df = pd.read_csv(core.SAMPLE_DIR / spec["file"])
    df = df.drop(columns=[c for c in spec["drop"] if c in df.columns])
    set_dataset(spec["file"], df.to_csv(index=False).encode(), spec["target"], spec["positive"])


@st.dialog("Choose a dataset", width="large")
def dataset_dialog() -> None:
    st.caption("Pick one of the built-in datasets, or upload your own CSV.")
    for spec in core.SAMPLE_DATASETS:
        with st.container(key=f"sample_{spec['file']}"):
            left, right = st.columns([3, 1], vertical_alignment="center")
            left.markdown(f"**{spec['title']}**  \n{spec['blurb']} Target: `{spec['target']}`.")
            if right.button("Use this", key=f"use_{spec['file']}", use_container_width=True):
                choose_sample(spec)
                st.rerun()
    st.markdown("**Or upload your own**")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        data = uploaded.getvalue()
        try:
            columns = inspect_csv(data)["columns"]
        except Exception as exc:
            st.error(f"Could not read this file as a CSV: {exc}")
        else:
            set_dataset(uploaded.name, data, columns[-1])
            st.rerun()


def sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="ws-label">WORKSPACE</div>', unsafe_allow_html=True)
        for name, slug in (("Setup", "setup"), ("Live run", "live"), ("Reports", "reports")):
            page = "Live Run" if slug == "live" else name
            active = st.session_state.page == page
            st.button(name, key=f"nav_{slug}_{'on' if active else 'off'}", on_click=go, args=(page,))
        with st.container(key="key_box"):
            api_key = st.text_input(
                "Anthropic API key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""),
                placeholder="sk-ant-…", help="Used only for your run. Never written to disk or shared.",
            )
            st.caption("Your key stays in this session and is passed only to your own run.")
        with st.container(key="upload_box"):
            if st.button("Upload Dataset", key="open_upload", use_container_width=True):
                dataset_dialog()
    return api_key


def page_setup(api_key: str):
    st.markdown("<h1>Setup</h1><div class='subtitle'>Start a new automated modeling run</div>", unsafe_allow_html=True)

    dataset = st.session_state.dataset
    info, data = None, b""
    if dataset is not None:
        data = dataset["data"]
        info = inspect_csv(data)

    with st.container(key="card_dataset"):
        st.markdown("<h3>Dataset &amp; Target</h3>", unsafe_allow_html=True)
        if info:
            size = len(data) / 1e6
            size_text = f"{size:.1f}mb" if size >= 0.1 else f"{len(data) / 1e3:.0f}kb"
            st.markdown(
                f'<div class="filebox"><div class="file">{FILE_ICON}<div><strong>{dataset['name']}</strong>'
                f'<span>{size_text} · {info["rows"]:,} rows · {len(info["columns"])} columns</span></div></div>{CHECK_ICON}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="filebox empty">Choose a dataset with <b>&nbsp;Upload Dataset&nbsp;</b> in the sidebar to begin.</div>',
                unsafe_allow_html=True,
            )
        c1, c2 = st.columns(2, gap="large")
        columns = info["columns"] if info else []
        target = c1.selectbox("Target column", columns, key="target_col",
                              placeholder="Choose a dataset first", disabled=not columns)
        seed = c2.number_input("Random seed", min_value=0, step=1, key="seed")

    with st.container(key="card_loop"):
        st.markdown("<h3>Loop Settings</h3>", unsafe_allow_html=True)
        with st.container(key="loop_box"):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.slider("Max Iterations", 1, 5, key="max_iterations")
                st.caption("Maximum number of iterations to run")
            with c2:
                st.slider("Target AUC-ROC", 0.50, 0.99, step=0.01, key="target_auc")
                st.caption("Stop when this performance target is reached")

    rows = info["rows"] if info else 7000
    lo, hi = core.estimate_minutes(rows, st.session_state.max_iterations)
    preset = json.dumps({
        "target_column": target, "seed": int(seed),
        "max_iterations": st.session_state.max_iterations, "performance_target": st.session_state.target_auc,
    }, indent=2)
    ready = bool(info and target and api_key)
    with st.container(key="card_run"):
        with st.container(key="runbar"):
            c_est, c_save, c_run = st.columns([1.3, 1, 1], gap="large", vertical_alignment="center")
            c_est.markdown(f'<div class="est">{CLOCK_ICON}<span>Est. {lo}-{hi} min</span></div>', unsafe_allow_html=True)
            c_save.download_button("Save Preset", data=preset, file_name="autods_preset.json",
                                   mime="application/json", use_container_width=True)
            run_clicked = c_run.button("Run Pipeline", type="primary", use_container_width=True, disabled=not ready)
        if not info:
            hint = "Choose a dataset to enable the run."
        elif not api_key:
            hint = "Add your Anthropic API key in the sidebar to enable the run."
        else:
            hint = "Each run calls the Anthropic API with your key; the estimate is approximate."
        st.markdown(f'<div class="note">{hint}</div>', unsafe_allow_html=True)
        with st.container(key="demo_link"):
            st.button("Or watch a recorded run (Telco churn, seed 42; no API key needed)",
                      key="replay_btn", on_click=lambda: start_demo(), type="tertiary")

    if run_clicked:
        try:
            if dataset["positive"] is not None and target == dataset["target"]:
                positive = dataset["positive"]
            else:
                positive = core.detect_positive_label(pd.read_csv(io.BytesIO(data), usecols=[target])[target])
        except ValueError as exc:
            st.error(str(exc))
            return
        if live_runs_active() >= MAX_CONCURRENT_LIVE_RUNS:
            st.error("The server is busy with other runs. Try again in a few minutes, or watch the recorded run.")
            return
        start_run("live", {
            "file_name": dataset["name"], "target_column": target, "positive_label": positive,
            "seed": int(seed), "max_iterations": st.session_state.max_iterations,
            "performance_target": st.session_state.target_auc,
        }, api_key=api_key, data=data)
        st.rerun()


def start_demo():
    demo = core.load_demo()
    start_run("replay", {
        "file_name": demo["file_name"], "target_column": demo["target_column"],
        "positive_label": demo["positive_label"], "seed": demo["config"]["seed"],
        "max_iterations": demo["config"]["max_iterations"],
        "performance_target": demo["config"]["performance_target"], "recorded": True,
    })


def live_body(run: dict):
    view = refresh_run(run)
    meta = run["meta"]
    state = view["state"] or initial_state(meta["max_iterations"])
    running = None
    if not run["finished"]:
        running = max(0.0, time.time() - view["last_ts"])
    if run["finished"] and view["state"] is None:
        state = initial_state(meta["max_iterations"])
    if meta.get("recorded"):
        st.markdown(
            '<div class="alert">Replaying a recorded run of the real pipeline (Telco churn, seed 42). '
            'Metrics and diagnoses are as logged; per-stage times are apportioned from the logged per-agent totals.</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        core.live_view_html(state, meta["file_name"], meta["seed"], meta["performance_target"], running),
        unsafe_allow_html=True,
    )
    with st.container(key="liveactions"):
        if run["error"]:
            st.markdown(f'<div class="alert error">{run["error"]}</div>', unsafe_allow_html=True)
        if run["result"] is not None and run["result"].get("used_fallback"):
            st.markdown(
                '<div class="alert">One or more agents fell back to default plans (the language-model call failed). '
                'If this was unexpected, check that your Anthropic API key is valid and has credit.</div>',
                unsafe_allow_html=True,
            )
        if not run["finished"]:
            if st.button("Stop run", key="stop_btn"):
                run["stopped"] = True
                kill_run(run)
                st.rerun()
        elif run["result"] is not None:
            if st.button("View Reports →", type="primary", key="to_reports"):
                go("Reports")
                st.rerun()
        elif st.button("Back to Setup", key="to_setup"):
            go("Setup")
            st.rerun()
    if run["finished"] and st.session_state.get("_polling", False):
        st.session_state["_polling"] = False
        st.rerun()


def page_live():
    st.markdown("<h1>Live Run</h1>", unsafe_allow_html=True)
    with st.container(key="backbtn"):
        st.button("Back", key="back_btn", on_click=go, args=("Setup",))
    run = st.session_state.run
    if run is None:
        st.markdown("<div class='subtitle'>No run in progress.</div>", unsafe_allow_html=True)
        st.button("Go to Setup", on_click=go, args=("Setup",), type="primary")
        return
    polling = not run["finished"]
    st.session_state["_polling"] = polling
    st.fragment(live_body, run_every=1.0 if polling else None)(run)


def page_reports():
    report = st.session_state.report
    st.markdown("<h1>Reports</h1>", unsafe_allow_html=True)
    if report is None:
        st.markdown("<div class='subtitle'>No completed run yet. Finish a run to see its report here.</div>", unsafe_allow_html=True)
        st.button("Go to Setup", on_click=go, args=("Setup",), type="primary")
        return
    result, meta = report["result"], report["meta"]
    points = core.result_points(result)
    if not points:
        st.warning("This run finished without any evaluated iteration.")
        return
    chosen = core.selected_entry(result) or {}
    tag = "recorded run" if meta.get("recorded") else f"seed {meta['seed']}"
    st.markdown(
        f'<div class="subline"><span class="meta">{core.DB_ICON}'
        f'{meta["file_name"]}<span class="dot">•</span>{tag}<span class="dot">•</span>'
        f'{result.get("iteration_count")} iterations</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="kpis-panel">{core.kpi_html(result, points)}</div>', unsafe_allow_html=True)
    stop = (result.get("extra_params") or {}).get("stop_reason", "")
    st.markdown(
        f'<div class="chart-card">{core.line_chart_svg(points)}'
        f'<div class="note">Reported model: iteration {chosen.get("iteration")} ({chosen.get("model")}), the highest-F1 '
        f'iteration within the AUC tolerance band: AUC-ROC {chosen.get("auc_roc")}, F1 {chosen.get("f1_weighted")}. '
        f'Loop stopped: {str(stop).replace("_", " ")}.</div></div>',
        unsafe_allow_html=True,
    )
    cache_key = f"pdf_{report['run_id']}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = core.build_pdf(result, meta["file_name"], meta["target_column"])
    config = json.dumps(
        core.model_config(result, meta["file_name"], meta["target_column"], meta["positive_label"]),
        indent=2, default=core.json_default,
    )
    stem = Path(meta["file_name"]).stem
    with st.container(key="reportbar"):
        c1, c2 = st.columns(2, gap="large")
        c1.download_button("Export as PDF", data=st.session_state[cache_key], file_name=f"autods_{stem}_report.pdf",
                           mime="application/pdf", use_container_width=True)
        c2.download_button("Download Model Config", data=config, file_name=f"autods_{stem}_model_config.json",
                           mime="application/json", type="primary", use_container_width=True)
    st.markdown(
        '<div class="note">Model Config is the reproducible specification of the selected pipeline '
        '(cleaning plan, feature plan, model, hyperparameters, threshold). AutoDS does not persist a fitted model object.</div>',
        unsafe_allow_html=True,
    )


live = st.session_state.page == "Live Run"
chrome(live=live)
api_key_value = sidebar()

if st.session_state.page == "Setup":
    page_setup(api_key_value)
elif live:
    page_live()
else:
    page_reports()
