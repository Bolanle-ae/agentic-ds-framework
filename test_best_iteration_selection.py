from pipeline import (
    BEST_ITERATION_AUC_TOLERANCE,
    _select_best_iteration,
    build_result_from_state,
    SIX_STAGES,
)


def _entry(iteration, auc, recall, precision):
    return {
        "iteration": iteration, "model": f"model_{iteration}", "auc_roc": auc,
        "recall_pos": recall, "precision_pos": precision, "threshold": 0.3,
        "accuracy": None, "f1_weighted": None, "train_auc_roc": None, "train_test_gap": None,
        "overfitting_flag": False, "underfitting_flag": False,
        "use_class_weighting": False, "class_weight_ratio": None, "calibration_metric_used": "f1",
    }


def test_real_credit_default_seed42_case():
    history = [
        _entry(1, 0.7746, 0.5177, 0.5444),
        _entry(2, 0.7793, 0.8402, 0.3199),
    ]
    picked = _select_best_iteration(history)
    assert picked["iteration"] == 1, f"expected iteration 1 (balanced), got {picked['iteration']}"
    print("PASS  test_real_credit_default_seed42_case (picks iter 1, not the flagged over-correction at iter 2)")


def test_real_credit_default_seed2024_case():
    history = [
        _entry(1, 0.7744, 0.5358, 0.5346),
        _entry(2, 0.7753, 0.5524, 0.5166),
        _entry(3, 0.7655, 0.6496, 0.4189),
        _entry(4, 0.7748, 0.8953, 0.2912),
        _entry(5, 0.7341, 0.5765, 0.4524),
    ]
    picked = _select_best_iteration(history)
    assert picked["iteration"] == 1, f"expected iteration 1, got {picked['iteration']}"
    assert picked["iteration"] != 4, "must never pick the degenerate iteration"
    print("PASS  test_real_credit_default_seed2024_case (avoids both the old pick and the degenerate iteration)")


def test_never_trades_away_real_auc_for_balance():
    history = [
        _entry(1, 0.70, 0.90, 0.85),
        _entry(2, 0.90, 0.50, 0.40),
    ]
    picked = _select_best_iteration(history)
    assert picked["iteration"] == 2, f"a real 0.20 AUC gap must not be overridden by balance, got iter {picked['iteration']}"
    print("PASS  test_never_trades_away_real_auc_for_balance")


def test_single_iteration_fallback():
    history = [_entry(1, 0.80, 0.5, 0.5)]
    picked = _select_best_iteration(history)
    assert picked["iteration"] == 1
    print("PASS  test_single_iteration_fallback")


def test_empty_history_returns_none():
    assert _select_best_iteration([]) is None
    print("PASS  test_empty_history_returns_none")


def test_no_change_when_max_auc_iteration_is_already_balanced():
    history = [
        _entry(1, 0.847, 0.714, 0.556),
        _entry(2, 0.842, 0.711, 0.557),
    ]
    picked = _select_best_iteration(history)
    assert picked["iteration"] == 1
    print("PASS  test_no_change_when_max_auc_iteration_is_already_balanced")


def test_tolerance_boundary_is_inclusive_and_exclusive_correctly():
    tol = BEST_ITERATION_AUC_TOLERANCE
    history_in = [_entry(1, 0.80, 0.9, 0.9), _entry(2, 0.80 + tol, 0.3, 0.2)]
    picked_in = _select_best_iteration(history_in)
    assert picked_in["iteration"] == 1, "an iteration exactly at the tolerance boundary should still be compared on F1"
    history_out = [_entry(1, 0.80, 0.9, 0.9), _entry(2, 0.80 + tol + 0.001, 0.3, 0.2)]
    picked_out = _select_best_iteration(history_out)
    assert picked_out["iteration"] == 2, "an iteration just outside tolerance must win on AUC alone"
    print("PASS  test_tolerance_boundary_is_inclusive_and_exclusive_correctly")


def test_full_integration_via_build_result_from_state():
    history = [
        _entry(1, 0.7746, 0.5177, 0.5444),
        _entry(2, 0.7793, 0.8402, 0.3199),
    ]
    final_state = {
        "iteration_history": history, "metrics": history[-1], "model_name": "model_2",
        "train_time_sec": 1.0, "stage_automation": {s: True for s in SIX_STAGES},
        "events": [], "used_fallback": False, "iteration": 2, "token_usage": {},
        "stage_timings": [], "optuna_durations": [], "stop_reason": "iteration_cap_reached",
        "threshold": 0.3, "use_class_weighting": False, "class_weight_ratio": None,
        "calibration_metric": "f1",
    }
    result = build_result_from_state(final_state, "credit_default", 42, max_iterations=5, optuna_trials=None, performance_target=0.85)
    assert result.auc_roc == 0.7746, f"expected iteration 1's AUC, got {result.auc_roc}"
    assert result.recall_pos == 0.5177
    assert result.precision_pos == 0.5444
    assert result.extra_params["best_iteration"] == 1
    assert result.extra_params["last_iteration"] == 2
    assert history[0]["selected_as_final"] is True
    assert history[1]["selected_as_final"] is False
    print("PASS  test_full_integration_via_build_result_from_state")


if __name__ == "__main__":
    test_real_credit_default_seed42_case()
    test_real_credit_default_seed2024_case()
    test_never_trades_away_real_auc_for_balance()
    test_single_iteration_fallback()
    test_empty_history_returns_none()
    test_no_change_when_max_auc_iteration_is_already_balanced()
    test_tolerance_boundary_is_inclusive_and_exclusive_correctly()
    test_full_integration_via_build_result_from_state()
    print("\nALL BEST-ITERATION SELECTION TESTS PASSED")
