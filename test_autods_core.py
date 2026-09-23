import pandas as pd

import autods_core as core


def _replayed_states():
    return list(core.replay_states(core.load_demo()))


def test_replay_reproduces_the_logged_iterations():
    demo = core.load_demo()
    final = _replayed_states()[-1]
    logged = demo["result"]["iteration_metrics"]
    assert [e["iteration"] for e in final["iteration_history"]] == [e["iteration"] for e in logged]
    assert [e["auc_roc"] for e in final["iteration_history"]] == [e["auc_roc"] for e in logged]
    assert final["stop_reason"] == "iteration_cap_reached"
    print("PASS  test_replay_reproduces_the_logged_iterations")


def test_agent_statuses_progress_and_finish():
    states = _replayed_states()
    first = core.compute_agent_statuses(states[0])
    assert first["profiler"][0] == "completed" and first["cleaner"][0] == "running"
    assert first["reflection"][0] == "pending"
    last = core.compute_agent_statuses(states[-1])
    assert all(status == "completed" for status, _ in last.values())
    print("PASS  test_agent_statuses_progress_and_finish")


def test_no_phantom_chart_point_between_iterations():
    for state in _replayed_states():
        iterations = [p["iteration"] for p in core.chart_points(state)]
        assert iterations == sorted(set(iterations)), iterations
        evaluated = sum(1 for t in state["stage_timings"] if t["stage"] == "evaluator")
        assert len(iterations) <= evaluated
    print("PASS  test_no_phantom_chart_point_between_iterations")


def test_positive_label_is_minority_class_and_binary_only():
    assert core.detect_positive_label(pd.Series(["No"] * 80 + ["Yes"] * 20, name="y")) == "Yes"
    assert core.detect_positive_label(pd.Series([1] * 5 + [0] * 95, name="y")) == 1
    try:
        core.detect_positive_label(pd.Series([0, 1, 2], name="y"))
    except ValueError:
        pass
    else:
        raise AssertionError("multiclass target should be rejected")
    print("PASS  test_positive_label_is_minority_class_and_binary_only")


def test_model_config_uses_the_reported_iteration():
    demo = core.load_demo()
    cfg = core.model_config(demo["result"], "telco_churn.csv", "Churn", "Yes")
    chosen = next(e for e in demo["result"]["iteration_metrics"] if e["selected_as_final"])
    assert cfg["selected_iteration"] == chosen["iteration"] and cfg["model"] == chosen["model"]
    assert cfg["best_params"] == chosen["best_params"]
    print("PASS  test_model_config_uses_the_reported_iteration")


def test_pdf_and_charts_render():
    demo = core.load_demo()
    pdf = core.build_pdf(demo["result"], "telco_churn.csv", "Churn")
    assert pdf[:4] == b"%PDF"
    points = core.result_points(demo["result"])
    assert "<svg" in core.line_chart_svg(points) and "<svg" in core.bar_chart_svg(points, 0.85)
    print("PASS  test_pdf_and_charts_render")


if __name__ == "__main__":
    test_replay_reproduces_the_logged_iterations()
    test_agent_statuses_progress_and_finish()
    test_no_phantom_chart_point_between_iterations()
    test_positive_label_is_minority_class_and_binary_only()
    test_model_config_uses_the_reported_iteration()
    test_pdf_and_charts_render()
    print("\nALL AUTODS CORE TESTS PASSED")
