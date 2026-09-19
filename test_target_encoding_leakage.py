import numpy as np
import pandas as pd

from pipeline import _target_encode_column, TARGET_ENCODING_FOLDS, TARGET_ENCODING_SMOOTHING


def test_no_self_leakage_unique_categories():
    n = 200
    seed = 42
    rng = np.random.RandomState(seed)
    y_train = rng.randint(0, 2, size=n)
    train_col = pd.Series([f"id_{i}" for i in range(n)])
    test_col = pd.Series([f"id_{i}" for i in range(n, n + 20)])

    train_encoded, test_encoded = _target_encode_column(train_col, test_col, y_train, seed)
    global_mean = y_train.mean()

    assert np.allclose(train_encoded, global_mean, atol=1e-9), (
        f"LEAK: train encodings are not all the global mean ({global_mean}); "
        f"got unique values {np.unique(train_encoded)} — a row's own label "
        f"reached its own encoded value."
    )
    assert np.allclose(test_encoded, global_mean, atol=1e-9), (
        "test encodings for never-seen categories should also fall back to "
        "the training global mean."
    )
    assert not np.allclose(train_encoded, y_train), (
        "SUSPICIOUS: encoded values equal the raw labels — this is exactly "
        "what naive (leaky) target encoding of unique categories looks like."
    )
    print("PASS  test_no_self_leakage_unique_categories")


def test_real_signal_is_captured():
    n = 400
    seed = 7
    rng = np.random.RandomState(seed)
    cats = rng.choice(["A", "B"], size=n)
    y = np.where(cats == "A", rng.binomial(1, 0.85, size=n), rng.binomial(1, 0.15, size=n))
    train_col = pd.Series(cats)
    test_col = pd.Series(["A", "B", "A", "B"])

    train_encoded, test_encoded = _target_encode_column(train_col, test_col, y, seed)

    mean_a = train_encoded[cats == "A"].mean()
    mean_b = train_encoded[cats == "B"].mean()
    assert mean_a > 0.6, f"category A should encode high (near its true 0.85 rate), got {mean_a}"
    assert mean_b < 0.4, f"category B should encode low (near its true 0.15 rate), got {mean_b}"
    assert test_encoded[0] > test_encoded[1], "test encoding must preserve A > B ordering"
    print(f"PASS  test_real_signal_is_captured (train: A={mean_a:.3f}, B={mean_b:.3f}; "
          f"test: A={test_encoded[0]:.3f}, B={test_encoded[1]:.3f})")


def test_train_encoding_independent_of_test_data():
    n = 150
    seed = 3
    rng = np.random.RandomState(seed)
    cats = rng.choice(["x", "y", "z"], size=n)
    y = rng.randint(0, 2, size=n)
    train_col = pd.Series(cats)

    test_col_a = pd.Series(["x", "y"])
    test_col_b = pd.Series(["z", "z", "z", "z", "z", "unseen_category"])

    train_encoded_a, _ = _target_encode_column(train_col, test_col_a, y, seed)
    train_encoded_b, _ = _target_encode_column(train_col, test_col_b, y, seed)

    assert np.array_equal(train_encoded_a, train_encoded_b), (
        "train_encoded changed when only test_col changed — test data is "
        "leaking into the training encoding."
    )
    print("PASS  test_train_encoding_independent_of_test_data")


if __name__ == "__main__":
    print(f"(TARGET_ENCODING_FOLDS={TARGET_ENCODING_FOLDS}, TARGET_ENCODING_SMOOTHING={TARGET_ENCODING_SMOOTHING})\n")
    test_no_self_leakage_unique_categories()
    test_real_signal_is_captured()
    test_train_encoding_independent_of_test_data()
    print("\nALL LEAKAGE TESTS PASSED")
