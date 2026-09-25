from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AGENT_ORDER = ["profiler", "cleaner", "feature_engineer", "model_selector", "evaluator", "reflection"]
AGENT_LABELS = {
    "profiler": "1. Data Profiling",
    "cleaner": "2. Data Cleaning",
    "feature_engineer": "3. Feature Engineering",
    "model_selector": "4. Model Selection",
    "evaluator": "5. Evaluation Agent",
    "reflection": "6. Reflection Agent",
}

HISTORY_KEYS = [
    "iteration", "model", "auc_roc", "f1_weighted", "recall_pos", "precision_pos",
    "train_test_gap", "diagnosis", "stop_reason",
]
METRIC_KEYS = ["auc_roc", "f1_weighted", "recall_pos", "precision_pos", "train_test_gap"]

DB_ICON = (
    '<svg viewBox="0 0 24 24" class="db" aria-hidden="true"><ellipse cx="12" cy="5.5" rx="8" ry="3.2" fill="#F5893F"/>'
    '<path d="M4 7.5v4c0 1.8 3.6 3.2 8 3.2s8-1.4 8-3.2v-4c0 1.8-3.6 3.2-8 3.2S4 9.3 4 7.5z" fill="#F5893F"/>'
    '<path d="M4 13.5v4c0 1.8 3.6 3.2 8 3.2s8-1.4 8-3.2v-4c0 1.8-3.6 3.2-8 3.2s-8-1.4-8-3.2z" fill="#F5893F"/></svg>'
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_DATASETS = [
    {"file": "telco_churn.csv", "title": "Telco Customer Churn", "target": "Churn", "positive": "Yes",
     "drop": ["customerID"], "blurb": "7,043 customers; will they churn?"},
    {"file": "credit_default.csv", "title": "Credit Card Default", "target": "default.payment.next.month", "positive": 1,
     "drop": [], "blurb": "30,000 cardholders; will they default next month?"},
    {"file": "stroke.csv", "title": "Stroke Prediction", "target": "stroke", "positive": 1,
     "drop": ["id"], "blurb": "5,110 patients; did they have a stroke?"},
]

DEMO_PATH = Path(__file__).parent / "demo" / "telco_churn_seed42.json"


def json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def slim_state(state: dict) -> dict:
    history = [{k: e.get(k) for k in HISTORY_KEYS} for e in state["iteration_history"]]
    metrics = state.get("metrics")
    return {
        "iteration": state["iteration"],
        "max_iterations": state["max_iterations"],
        "stage_timings": list(state["stage_timings"]),
        "iteration_history": history,
        "metrics": {k: metrics.get(k) for k in METRIC_KEYS} if metrics else None,
        "stop_reason": state.get("stop_reason"),
        "weakest_component": state.get("weakest_component"),
        "current_failure_mode": state.get("current_failure_mode"),
        "current_instruction": state.get("current_instruction"),
    }


def compute_agent_statuses(state: dict) -> Dict[str, Tuple[str, Optional[float]]]:
    stopped = state.get("stop_reason") is not None
    current_iter = state["iteration"] if stopped else state["iteration"] + 1

    counts: Dict[str, int] = {}
    last_duration: Dict[str, float] = {}
    for entry in state["stage_timings"]:
        counts[entry["stage"]] = counts.get(entry["stage"], 0) + 1
        last_duration[entry["stage"]] = entry["duration_sec"]

    statuses: Dict[str, Tuple[str, Optional[float]]] = {}
    found_running = False
    for agent in AGENT_ORDER:
        expected = 1 if agent == "profiler" else max(current_iter, 1)
        if counts.get(agent, 0) >= expected:
            statuses[agent] = ("completed", last_duration.get(agent))
        elif not found_running and not stopped:
            statuses[agent] = ("running", None)
            found_running = True
        else:
            statuses[agent] = ("pending", None)
    return statuses


def chart_points(state: dict) -> List[dict]:
    rows = [
        {"iteration": e["iteration"], "auc": e.get("auc_roc"), "f1": e.get("f1_weighted")}
        for e in state["iteration_history"]
    ]
    metrics = state.get("metrics")
    current_iter = state["iteration"] + 1
    evaluated = sum(1 for t in state["stage_timings"] if t["stage"] == "evaluator")
    if metrics and not state.get("stop_reason") and evaluated >= current_iter and (not rows or rows[-1]["iteration"] != current_iter):
        rows.append({"iteration": current_iter, "auc": metrics.get("auc_roc"), "f1": metrics.get("f1_weighted")})
    return [r for r in rows if r["auc"] is not None]


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-- --"
    if seconds < 1:
        return "<1s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def estimate_minutes(rows: int, max_iterations: int) -> Tuple[int, int]:
    per_iteration_sec = 20 + 2 * rows / 1000
    total = per_iteration_sec * max_iterations
    return max(1, int(total * 0.5 / 60)), max(2, math.ceil(total * 1.5 / 60))


def detect_positive_label(series) -> Any:
    counts = series.dropna().value_counts()
    if len(counts) != 2:
        raise ValueError(
            f"Target column '{series.name}' has {len(counts)} distinct values; AutoDS handles binary targets only."
        )
    if counts.iloc[0] == counts.iloc[1]:
        label = sorted(counts.index, key=str)[-1]
    else:
        label = counts.index[-1]
    return label.item() if hasattr(label, "item") else label


def _nice_axis(lo: float, hi: float) -> Tuple[float, float, float]:
    lo = max(0.0, math.floor((lo - 0.02) * 20) / 20)
    hi = min(1.0, math.ceil((hi + 0.02) * 20) / 20)
    if hi - lo < 0.2:
        hi = min(1.0, lo + 0.2)
    step = 0.05 if hi - lo <= 0.3 else 0.1
    return lo, hi, step


def sparkline_svg(values: List[float], width: int = 300, height: int = 70, color: str = "#5FAE7B") -> str:
    values = [v for v in values if v is not None]
    if not values:
        return f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" aria-label="no data"></svg>'
    pad = 6
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(values)

    def xy(i: int, v: float) -> Tuple[float, float]:
        x = pad + (width - 2 * pad) * (i / (n - 1) if n > 1 else 0.5)
        y = height - pad - (height - 2 * pad) * ((v - lo) / span if hi != lo else 0.5)
        return x, y

    pts = [xy(i, v) for i, v in enumerate(values)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    last_x, last_y = pts[-1]
    line = f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>' if n > 1 else ""
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" aria-label="trend">'
        f'{line}<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4.5" fill="{color}"/></svg>'
    )


def bar_chart_svg(points: List[dict], target: Optional[float]) -> str:
    width, height = 620, 260
    left, right, top, bottom = 52, 78, 26, 40
    values = [p["auc"] for p in points]
    lo = 0.4 if min(values) >= 0.45 else max(0.0, math.floor(min(values) * 10) / 10 - 0.1)
    hi = min(1.0, max(0.9, math.ceil((max(values + [target or 0.0]) + 0.02) * 10) / 10))
    plot_w, plot_h = width - left - right, height - top - bottom

    def y_of(v: float) -> float:
        return top + plot_h * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="AUC-ROC by iteration">']
    ticks = int(round((hi - lo) / 0.1))
    for i in range(ticks + 1):
        v = lo + i * 0.1
        parts.append(f'<text x="{left - 8}" y="{y_of(v) + 4:.1f}" text-anchor="end" class="axis">{v:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis-line"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis-line"/>')
    slots = max(len(points), 3)
    slot_w = plot_w / slots
    bar_w = min(slot_w * 0.62, 120)
    for i, p in enumerate(points):
        cx = left + slot_w * (i + 0.5)
        y = y_of(max(p["auc"], lo))
        parts.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" '
            f'rx="3" fill="#F5A05C"><title>iter {p["iteration"]}: AUC-ROC {p["auc"]:.4f}</title></rect>'
        )
        parts.append(f'<text x="{cx:.1f}" y="{y - 7:.1f}" text-anchor="middle" class="value">{p["auc"]:.3f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 20}" text-anchor="middle" class="axis">iter {p["iteration"]}</text>')
    if target is not None and lo <= target <= hi:
        ty = y_of(target)
        parts.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{width - right}" y2="{ty:.1f}" class="target-line"/>')
        parts.append(f'<text x="{width - right + 8}" y="{ty + 4:.1f}" class="axis">target {target:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart_svg(points: List[dict]) -> str:
    width, height = 860, 330
    left, right, top, bottom = 64, 40, 20, 62
    series = [("AUC-ROC", "auc", "#F5893F"), ("F1 Score", "f1", "#3F9A4E")]
    all_values = [p[key] for p in points for _, key, _ in series if p.get(key) is not None]
    lo, hi, step = _nice_axis(min(all_values), max(all_values))
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(points)

    def x_of(i: int) -> float:
        return left + plot_w * ((i + 0.5) / n)

    def y_of(v: float) -> float:
        return top + plot_h * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="AUC-ROC and F1 by iteration">']
    ticks = int(round((hi - lo) / step))
    for i in range(ticks + 1):
        v = lo + i * step
        parts.append(f'<line x1="{left}" y1="{y_of(v):.1f}" x2="{width - right}" y2="{y_of(v):.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y_of(v) + 4:.1f}" text-anchor="end" class="axis">{v:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis-line"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis-line"/>')
    for i, p in enumerate(points):
        parts.append(f'<text x="{x_of(i):.1f}" y="{top + plot_h + 20}" text-anchor="middle" class="axis">{p["iteration"]}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 38}" text-anchor="middle" class="axis">Iterations</text>')
    for name, key, color in series:
        pts = [(x_of(i), y_of(p[key])) for i, p in enumerate(points) if p.get(key) is not None]
        if len(pts) > 1:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for (x, y), p in zip(pts, [q for q in points if q.get(key) is not None]):
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#F0EDEB" stroke-width="2">'
                f'<title>{name}, iter {p["iteration"]}: {p[key]:.4f}</title></circle>'
            )
    legend_y = height - 8
    for j, (name, _, color) in enumerate(series):
        lx = left + plot_w * (0.25 + 0.4 * j)
        parts.append(f'<line x1="{lx:.1f}" y1="{legend_y - 4}" x2="{lx + 36:.1f}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 46:.1f}" y="{legend_y}" class="axis">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _signed(delta: float) -> Tuple[str, str]:
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "")
    cls = "up" if delta > 0 else ("down" if delta < 0 else "flat")
    return f"{arrow}{abs(delta):.3f}" if arrow else "±0.000", cls


def _card(agent: str, status: str, duration: Optional[float], elapsed: Optional[float]) -> str:
    label = AGENT_LABELS[agent]
    number, _, name = label.partition(". ")
    if status == "completed":
        icon = '<svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" fill="#3B9A4A"/><path d="M6.5 12.5l3.8 3.8 7.2-7.6" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        state_line = f'Completed {icon}'
        time_line = format_duration(duration)
    elif status == "running":
        icon = '<svg class="ic spin" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9.5" fill="none" stroke="#F5893F" stroke-width="2.6" stroke-dasharray="40 20" stroke-linecap="round"/></svg>'
        state_line = f'Running {icon}'
        time_line = format_duration(elapsed)
    else:
        icon = '<svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" fill="#C9CACC"/><path d="M12 6v6.5l3.5 2" fill="none" stroke="#5B5D60" stroke-width="2.4" stroke-linecap="round"/></svg>'
        state_line = "Pending"
        time_line = "-- --"
    return (
        f'<div class="agent {status}"><div class="agent-head">{icon}<span>{number}. {html.escape(name)}</span></div>'
        f'<div class="agent-state">{state_line}</div><div class="agent-time">{time_line}</div></div>'
    )


def live_view_html(state: dict, file_name: str, seed: int, target: float, elapsed_running: Optional[float]) -> str:
    statuses = compute_agent_statuses(state)
    cards = "".join(
        _card(a, statuses[a][0], statuses[a][1], elapsed_running if statuses[a][0] == "running" else None)
        for a in AGENT_ORDER
    )
    points = chart_points(state)
    aucs = [p["auc"] for p in points]
    f1s = [p["f1"] for p in points if p["f1"] is not None]

    def tile(title: str, values: List[float]) -> str:
        if not values:
            return f'<div class="tile"><h4>{title}</h4><div class="tile-body"><div class="big">—</div></div></div>'
        delta_html = ""
        if len(values) > 1:
            text, cls = _signed(values[-1] - values[-2])
            delta_html = f'<span class="delta {cls}">{text}</span>'
        return (
            f'<div class="tile"><h4>{title}</h4><div class="tile-body"><div class="tile-top">'
            f'<div class="big">{values[-1]:.3f}</div>{delta_html}</div>{sparkline_svg(values)}</div></div>'
        )

    chart = bar_chart_svg(points, target) if points else '<p class="muted">Bars appear once the first iteration is evaluated.</p>'

    history = state["iteration_history"]
    if history:
        latest = history[-1]
        diagnosis = latest.get("diagnosis") or {}
        head = f'Reflection <span class="dot">•</span> Iteration {latest["iteration"]}'
        body = (
            f'<h5>Diagnosis</h5><p>{html.escape(diagnosis.get("failure_mode") or "—")}</p>'
            f'<h5>Instruction for next attempt</h5><p>{html.escape(diagnosis.get("instruction") or "—")}</p>'
        )
        if state.get("stop_reason"):
            body += f'<p class="muted">Loop stopped ({html.escape(state["stop_reason"].replace("_", " "))}); this instruction was not applied.</p>'
    else:
        head = "Reflection"
        body = '<p class="muted">The reflection agent reports here after the first iteration is evaluated.</p>'

    current_iter = state["iteration"] if state.get("stop_reason") else state["iteration"] + 1
    meta = (
        f'<span class="meta">{DB_ICON}{html.escape(file_name)}<span class="dot">•</span>Random seed {seed}'
        f'<span class="dot">•</span>Iteration {min(current_iter, state["max_iterations"])} of {state["max_iterations"]}</span>'
    )
    return (
        f'<div class="lr-meta">{meta}</div>'
        f'<section class="panel pipeline"><h3>Agent Pipeline</h3><div class="agents">{cards}</div></section>'
        f'<div class="lr-grid"><div class="lr-left"><div class="tiles">{tile("AUC-ROC", aucs)}{tile("F1 Score", f1s)}</div>'
        f'<section class="panel chart-panel">{chart}</section></div>'
        f'<section class="panel reflect"><div class="reflect-inner"><h3>{head}</h3>{body}</div></section></div>'
    )


def kpi_html(result: dict, points: List[dict]) -> str:
    best = max(points, key=lambda p: p["auc"])
    first = points[0]
    auc_delta = best["auc"] - first["auc"]
    f1_points = [p for p in points if p["f1"] is not None]
    best_f1 = max(f1_points, key=lambda p: p["f1"]) if f1_points else None

    def sub(delta: float, iteration: int) -> str:
        if iteration == points[0]["iteration"] or abs(delta) < 0.0005:
            return f'<span>iter {iteration}</span>'
        return f'<span class="gain">+{delta:.3f}</span><span>iter {iteration}</span>'

    cost = result.get("api_token_cost_usd") or 0.0
    cost_text = f"{cost:.3f}" if cost < 1 else f"{cost:.2f}"
    runtime = format_duration(result.get("runtime_wallclock_sec") or 0.0).replace("<1s", "00:00")
    f1_tile = (
        f'<div class="kpi"><em>Best F1 Score</em><strong>{best_f1["f1"]:.3f}</strong><div class="sub">{sub(best_f1["f1"] - f1_points[0]["f1"], best_f1["iteration"])}</div></div>'
        if best_f1 else '<div class="kpi"><em>Best F1 Score</em><strong>—</strong></div>'
    )
    return (
        '<div class="kpis">'
        f'<div class="kpi"><em>Best AUC-ROC</em><strong>{best["auc"]:.3f}</strong><div class="sub">{sub(auc_delta, best["iteration"])}</div></div>'
        f'{f1_tile}'
        f'<div class="kpi"><em>Total API cost</em><strong>${cost_text}</strong></div>'
        f'<div class="kpi"><em>Total runtime</em><strong>{runtime}</strong></div></div>'
    )


def result_points(result: dict) -> List[dict]:
    return [
        {"iteration": e["iteration"], "auc": e.get("auc_roc"), "f1": e.get("f1_weighted")}
        for e in (result.get("iteration_metrics") or [])
        if e.get("auc_roc") is not None
    ]


def selected_entry(result: dict) -> Optional[dict]:
    for e in result.get("iteration_metrics") or []:
        if e.get("selected_as_final"):
            return e
    entries = result.get("iteration_metrics") or []
    return entries[-1] if entries else None


def model_config(result: dict, file_name: str, target_column: str, positive_label: Any) -> dict:
    entry = selected_entry(result) or {}
    return {
        "dataset": file_name,
        "target_column": target_column,
        "positive_label": positive_label,
        "seed": result.get("seed"),
        "selected_iteration": entry.get("iteration"),
        "model": entry.get("model"),
        "best_params": entry.get("best_params"),
        "threshold": entry.get("threshold"),
        "class_weighting": {
            "enabled": entry.get("use_class_weighting"),
            "ratio": entry.get("class_weight_ratio"),
        },
        "threshold_calibration_metric": entry.get("calibration_metric_used"),
        "cleaning_plan": entry.get("cleaning_plan"),
        "feature_plan": entry.get("feature_plan"),
        "final_feature_count": entry.get("final_feature_count"),
        "test_metrics": {
            k: entry.get(k) for k in ("auc_roc", "f1_weighted", "recall_pos", "precision_pos", "accuracy")
        },
        "note": "Reproducible specification of the selected pipeline. AutoDS does not persist a fitted model object.",
    }


_PDF_SUBSTITUTIONS = str.maketrans({"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "→": "->", "≥": ">=", "≤": "<=", "…": "..."})


def _latin(text: Any) -> str:
    return str(text).translate(_PDF_SUBSTITUTIONS).encode("latin-1", "replace").decode("latin-1")


def build_pdf(result: dict, file_name: str, target_column: str) -> bytes:
    from fpdf import FPDF

    points = result_points(result)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    orange = (245, 137, 63)

    pdf.set_fill_color(*orange)
    pdf.rect(0, 0, 210, 6, "F")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_y(14)
    pdf.cell(0, 10, "AutoDS run report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, _latin(f"{file_name}  |  target: {target_column}  |  seed {result.get('seed')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    best = max(points, key=lambda p: p["auc"]) if points else None
    f1_values = [p for p in points if p["f1"] is not None]
    best_f1 = max(f1_values, key=lambda p: p["f1"]) if f1_values else None
    kpis = [
        ("Best AUC-ROC", f'{best["auc"]:.3f} (iter {best["iteration"]})' if best else "-"),
        ("Best F1 (weighted)", f'{best_f1["f1"]:.3f} (iter {best_f1["iteration"]})' if best_f1 else "-"),
        ("Total API cost", f'${(result.get("api_token_cost_usd") or 0):.4f}'),
        ("Total runtime", format_duration(result.get("runtime_wallclock_sec") or 0).replace("<1s", "00:00")),
    ]
    box_w = 46
    y0 = pdf.get_y()
    for i, (label, value) in enumerate(kpis):
        x = 10 + i * (box_w + 2)
        pdf.set_fill_color(240, 237, 235)
        pdf.rect(x, y0, box_w, 20, "F")
        pdf.set_xy(x + 2, y0 + 3)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(box_w - 4, 4, label)
        pdf.set_xy(x + 2, y0 + 10)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(box_w - 4, 6, _latin(value))
    pdf.set_xy(pdf.l_margin, y0 + 26)

    entry = selected_entry(result)
    if entry:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(
            0, 5,
            _latin(
                f"Reported model: iteration {entry['iteration']} ({entry.get('model')}), selected as the highest-F1 "
                f"iteration within the AUC tolerance band. AUC-ROC {entry.get('auc_roc'):.4f}, "
                f"F1 {entry.get('f1_weighted'):.4f}, recall {entry.get('recall_pos'):.4f}, "
                f"precision {entry.get('precision_pos'):.4f}."
            ),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.ln(3)

    if len(points) > 1:
        cx, cy, cw, ch = 24, pdf.get_y() + 4, 170, 60
        vals = [p[k] for p in points for k in ("auc", "f1") if p.get(k) is not None]
        lo, hi, step = _nice_axis(min(vals), max(vals))
        pdf.set_draw_color(220, 220, 220)
        pdf.set_font("Helvetica", "", 7)
        ticks = int(round((hi - lo) / step))
        for i in range(ticks + 1):
            v = lo + i * step
            y = cy + ch * (1 - (v - lo) / (hi - lo))
            pdf.line(cx, y, cx + cw, y)
            pdf.set_xy(cx - 12, y - 2)
            pdf.cell(10, 4, f"{v:.2f}", align="R")
        n = len(points)
        for name, key, rgb in (("AUC-ROC", "auc", orange), ("F1", "f1", (63, 154, 78))):
            pdf.set_draw_color(*rgb)
            pdf.set_line_width(0.7)
            xy = [
                (cx + cw * ((i + 0.5) / n), cy + ch * (1 - (p[key] - lo) / (hi - lo)))
                for i, p in enumerate(points) if p.get(key) is not None
            ]
            for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
                pdf.line(x1, y1, x2, y2)
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        for i, p in enumerate(points):
            pdf.set_xy(cx + cw * ((i + 0.5) / n) - 5, cy + ch + 1)
            pdf.cell(10, 4, str(p["iteration"]), align="C")
        pdf.set_xy(cx, cy + ch + 6)
        pdf.set_text_color(*orange)
        pdf.cell(30, 4, "- AUC-ROC")
        pdf.set_text_color(63, 154, 78)
        pdf.cell(30, 4, "- F1 (weighted)")
        pdf.set_text_color(0, 0, 0)
        pdf.set_xy(pdf.l_margin, cy + ch + 14)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Per-iteration trail", new_x="LMARGIN", new_y="NEXT")
    for e in result.get("iteration_metrics") or []:
        pdf.set_font("Helvetica", "B", 10)
        marker = "  (reported)" if e.get("selected_as_final") else ""
        pdf.cell(
            0, 6,
            _latin(f"Iteration {e['iteration']} - {e.get('model')}{marker}"),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(
            0, 5,
            _latin(
                f"AUC {e.get('auc_roc')}  F1 {e.get('f1_weighted')}  recall {e.get('recall_pos')}  "
                f"precision {e.get('precision_pos')}  threshold {e.get('threshold')}"
            ),
            new_x="LMARGIN", new_y="NEXT",
        )
        d = e.get("diagnosis") or {}
        if d.get("failure_mode"):
            pdf.multi_cell(0, 4.5, _latin(f"Diagnosis: {d['failure_mode']}"), new_x="LMARGIN", new_y="NEXT")
        if d.get("instruction") and not e.get("stop_reason"):
            pdf.multi_cell(0, 4.5, _latin(f"Next instruction: {d['instruction']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    out = pdf.output()
    return bytes(out)


def load_demo() -> dict:
    return json.loads(DEMO_PATH.read_text())


def replay_states(demo: dict):
    result = demo["result"]
    iterations = result["iteration_metrics"]
    n = len(iterations)
    totals = demo["stage_seconds_total"]
    optuna = demo.get("optuna_seconds_per_call") or []
    optuna_sum = sum(optuna) or 1.0
    max_iterations = int(demo["config"]["max_iterations"])

    def share(agent: str, i: int) -> float:
        if agent == "profiler":
            return totals[agent]
        if agent == "model_selector" and len(optuna) == n:
            return totals[agent] * optuna[i] / optuna_sum
        return totals[agent] / n

    timings: List[dict] = []
    history: List[dict] = []
    for i, entry in enumerate(iterations):
        for agent in AGENT_ORDER:
            if agent == "profiler" and i > 0:
                continue
            timings.append({"stage": agent, "duration_sec": share(agent, i)})
            metrics = {k: entry.get(k) for k in METRIC_KEYS}
            if agent == "reflection":
                history.append({k: entry.get(k) for k in HISTORY_KEYS})
            has_metrics = agent in ("evaluator", "reflection")
            latest = history[-1] if history else {}
            yield {
                "iteration": len(history),
                "max_iterations": max_iterations,
                "stage_timings": list(timings),
                "iteration_history": list(history),
                "metrics": metrics if has_metrics else (
                    {k: history[-1].get(k) for k in METRIC_KEYS} if history else None
                ),
                "stop_reason": latest.get("stop_reason") if agent == "reflection" else None,
                "weakest_component": (latest.get("diagnosis") or {}).get("weakest_component"),
                "current_failure_mode": (latest.get("diagnosis") or {}).get("failure_mode"),
                "current_instruction": (latest.get("diagnosis") or {}).get("instruction"),
            }
