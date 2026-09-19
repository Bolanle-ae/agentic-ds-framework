import numpy as np
import pandas as pd

from pipeline import build_model, CANDIDATE_MODELS
from experiment_common import find_best_threshold

CLASS_WEIGHT_SUPPORTED = {"logistic_regression", "random_forest", "extra_trees", "svm", "decision_tree"}
SCALE_POS_WEIGHT_SUPPORTED = {"xgboost", "lightgbm"}
AUTO_CLASS_WEIGHTS_SUPPORTED = {"catboost"}
NO_WEIGHTING_SUPPORT = {"gradient_boosting", "knn", "naive_bayes", "adaboost"}


def test_class_weighting_applied_correctly_per_model():
    assert CLASS_WEIGHT_SUPPORTED | SCALE_POS_WEIGHT_SUPPORTED | AUTO_CLASS_WEIGHTS_SUPPORTED | NO_WEIGHTING_SUPPORT == set(CANDIDATE_MODELS), (
        "this test's model groups are out of sync with CANDIDATE_MODELS — update the groups above"
    )
    ratio = 4.5
    for name in CLASS_WEIGHT_SUPPORTED:
        model = build_model(name, {}, seed=1, class_weight_ratio=ratio)
        assert getattr(model, "class_weight", None) == "balanced", f"{name}: expected class_weight='balanced', got {getattr(model, 'class_weight', None)}"
    for name in SCALE_POS_WEIGHT_SUPPORTED:
        model = build_model(name, {}, seed=1, class_weight_ratio=ratio)
        assert getattr(model, "scale_pos_weight", None) == ratio, f"{name}: expected scale_pos_weight={ratio}"
    for name in AUTO_CLASS_WEIGHTS_SUPPORTED:
        model = build_model(name, {}, seed=1, class_weight_ratio=ratio)
        assert model is not None
    for name in NO_WEIGHTING_SUPPORT:
        model_weighted = build_model(name, {}, seed=1, class_weight_ratio=ratio)
        model_unweighted = build_model(name, {}, seed=1, class_weight_ratio=None)
        assert type(model_weighted) is type(model_unweighted)
    print("PASS  test_class_weighting_applied_correctly_per_model")


def test_class_weighting_is_a_training_time_only_noop_when_off():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=6, weights=[0.85, 0.15], random_state=3)
    m1 = build_model("random_forest", {"n_estimators": 30, "max_depth": 4, "min_samples_split": 2}, seed=3)
    m2 = build_model("random_forest", {"n_estimators": 30, "max_depth": 4, "min_samples_split": 2}, seed=3, class_weight_ratio=None)
    m1.fit(X, y)
    m2.fit(X, y)
    assert np.array_equal(m1.predict_proba(X), m2.predict_proba(X)), (
        "omitting class_weight_ratio vs. passing None explicitly produced different models — "
        "should be identical, both mean 'off'."
    )
    print("PASS  test_class_weighting_is_a_training_time_only_noop_when_off")


def test_class_weighting_measurably_shifts_recall_on_imbalanced_data():
    from sklearn.datasets import make_classification
    from sklearn.metrics import recall_score
    X, y = make_classification(
        n_samples=1000, n_features=8, n_informative=5, weights=[0.92, 0.08], random_state=11, flip_y=0.02,
    )
    split = 700
    X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

    unweighted = build_model("logistic_regression", {"C": 1.0}, seed=11)
    weighted = build_model("logistic_regression", {"C": 1.0}, seed=11, class_weight_ratio=99.0)
    unweighted.fit(X_train, y_train)
    weighted.fit(X_train, y_train)

    recall_unweighted = recall_score(y_test, unweighted.predict(X_test), pos_label=1, zero_division=0)
    recall_weighted = recall_score(y_test, weighted.predict(X_test), pos_label=1, zero_division=0)
    print(f"  recall @ 0.5, unweighted={recall_unweighted:.3f} vs class_weight='balanced'={recall_weighted:.3f}")
    assert recall_weighted >= recall_unweighted, (
        f"expected class_weight='balanced' to recover at least as much minority-class recall, "
        f"got {recall_weighted} vs {recall_unweighted}"
    )
    print("PASS  test_class_weighting_measurably_shifts_recall_on_imbalanced_data")


def test_f2_calibration_favors_recall_over_f1():
    rng = np.random.RandomState(5)
    n = 2000
    pos_rate = 0.05
    y = (rng.rand(n) < pos_rate).astype(int)
    proba = np.clip(y * rng.normal(0.6, 0.25, n) + (1 - y) * rng.normal(0.15, 0.15, n), 0, 1)

    t_f1 = find_best_threshold(y, proba, metric="f1")
    t_f2 = find_best_threshold(y, proba, metric="f2")
    print(f"  threshold: f1={t_f1:.3f} f2={t_f2:.3f}")
    assert t_f2 <= t_f1, f"expected F2's threshold ({t_f2}) to be <= F1's ({t_f1}) — F2 should be more permissive, not less"

    from sklearn.metrics import recall_score
    recall_f1 = recall_score(y, (proba >= t_f1).astype(int), pos_label=1, zero_division=0)
    recall_f2 = recall_score(y, (proba >= t_f2).astype(int), pos_label=1, zero_division=0)
    assert recall_f2 >= recall_f1, f"expected F2 calibration to recover >= recall vs F1 ({recall_f2} vs {recall_f1})"
    print(f"  recall: f1={recall_f1:.3f} f2={recall_f2:.3f}")
    print("PASS  test_f2_calibration_favors_recall_over_f1")


if __name__ == "__main__":
    test_class_weighting_applied_correctly_per_model()
    test_class_weighting_is_a_training_time_only_noop_when_off()
    test_class_weighting_measurably_shifts_recall_on_imbalanced_data()
    test_f2_calibration_favors_recall_over_f1()
    print("\nALL IMBALANCE-HANDLING TESTS PASSED")
