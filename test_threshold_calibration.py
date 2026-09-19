import inspect

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score

from experiment_common import calibrate_threshold_oof, find_best_threshold


def test_signature_has_no_test_data_parameter():
    params = list(inspect.signature(calibrate_threshold_oof).parameters)
    forbidden = {"X_test", "y_test", "test_col", "X_test_features"}
    leaked = forbidden & set(params)
    assert not leaked, f"LEAK: calibrate_threshold_oof accepts test data via {leaked}"
    assert set(params) == {
        "estimator", "X_train", "y_train", "seed", "metric",
        "subsample_row_threshold", "subsample_size",
    }, f"unexpected signature: {params}"
    print("PASS  test_signature_has_no_test_data_parameter")


def test_out_of_fold_not_in_sample():
    n = 400
    seed = 11
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 30)
    y = (X[:, 0] + rng.randn(n) * 0.5 > 0).astype(int)

    tree = DecisionTreeClassifier(random_state=seed)
    tree.fit(X, y)
    in_sample_proba = tree.predict_proba(X)[:, 1]
    in_sample_acc_like = np.mean((in_sample_proba > 0.5).astype(int) == y)

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(
        DecisionTreeClassifier(random_state=seed), X, y, cv=cv, method="predict_proba"
    )[:, 1]
    oof_acc_like = np.mean((oof_proba > 0.5).astype(int) == y)

    assert in_sample_acc_like > 0.99, (
        f"test setup issue: expected the unconstrained tree to memorize "
        f"training data in-sample, got {in_sample_acc_like}"
    )
    assert oof_acc_like < in_sample_acc_like - 0.05, (
        f"SUSPICIOUS: out-of-fold accuracy ({oof_acc_like:.3f}) is as high as "
        f"in-sample ({in_sample_acc_like:.3f}) — this looks like a model that "
        f"saw its own training rows before scoring them, not genuine OOF."
    )
    print(f"PASS  test_out_of_fold_not_in_sample (in-sample~{in_sample_acc_like:.3f} "
          f"vs out-of-fold~{oof_acc_like:.3f}, confirming the fold split is real)")


def test_calibration_recovers_recall_on_imbalanced_data():
    n = 2000
    seed = 5
    rng = np.random.RandomState(seed)
    pos_rate = 0.05
    y = (rng.rand(n) < pos_rate).astype(int)
    X = np.column_stack([
        y * rng.normal(1.5, 1.0, n) + (1 - y) * rng.normal(0.0, 1.0, n),
        rng.randn(n),
    ])

    split = int(n * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]

    naive_recall = recall_score(y_test, (proba_test >= 0.5).astype(int), pos_label=1, zero_division=0)

    threshold = calibrate_threshold_oof(
        RandomForestClassifier(random_state=seed), X_train, y_train, seed, metric="f1",
    )
    calibrated_recall = recall_score(y_test, (proba_test >= threshold).astype(int), pos_label=1, zero_division=0)

    print(f"  naive threshold=0.5 recall={naive_recall:.3f} | "
          f"calibrated threshold={threshold:.3f} recall={calibrated_recall:.3f}")
    assert threshold < 0.5, f"expected calibration to lower the threshold on imbalanced data, got {threshold}"
    assert calibrated_recall > naive_recall, (
        f"calibration should recover more positive-class recall than the naive "
        f"0.5 cutoff on this imbalanced setup ({calibrated_recall} vs {naive_recall})"
    )
    print("PASS  test_calibration_recovers_recall_on_imbalanced_data")


def test_find_best_threshold_is_pure_and_test_data_free():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 200)
    proba = rng.rand(200)
    t1 = find_best_threshold(y, proba, metric="f1")
    t2 = find_best_threshold(y, proba, metric="f1")
    assert t1 == t2, "find_best_threshold is not deterministic for identical inputs"
    assert 0.0 <= t1 <= 1.0
    print(f"PASS  test_find_best_threshold_is_pure_and_test_data_free (threshold={t1:.3f})")


if __name__ == "__main__":
    test_signature_has_no_test_data_parameter()
    test_out_of_fold_not_in_sample()
    test_calibration_recovers_recall_on_imbalanced_data()
    test_find_best_threshold_is_pure_and_test_data_free()
    print("\nALL THRESHOLD CALIBRATION TESTS PASSED")
