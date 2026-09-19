import numpy as np
import pandas as pd

from pipeline import (
    _binary_encode_columns,
    _count_encode_column,
    _ordinal_encode_column,
    ORDINAL_ENCODING_FOLDS,
)


def test_ordinal_no_self_leakage_unique_categories():
    n = 200
    seed = 42
    rng = np.random.RandomState(seed)
    y_train = rng.randint(0, 2, size=n)
    train_col = pd.Series([f"id_{i}" for i in range(n)])
    test_col = pd.Series([f"id_{i}" for i in range(n, n + 20)])

    train_encoded, test_encoded = _ordinal_encode_column(train_col, test_col, y_train, seed)

    assert len(np.unique(train_encoded)) == 1, (
        f"LEAK: unique-category train rows got different ranks ({np.unique(train_encoded)}) — "
        f"a row's own label reached its own encoded value."
    )
    assert len(np.unique(test_encoded)) == 1, "unseen test categories should also all fall back to the same rank"
    print("PASS  test_ordinal_no_self_leakage_unique_categories")


def test_ordinal_real_signal_is_captured():
    n = 400
    seed = 7
    rng = np.random.RandomState(seed)
    cats = rng.choice(["A", "B", "C"], size=n)
    y = np.select(
        [cats == "A", cats == "B", cats == "C"],
        [rng.binomial(1, 0.85, size=n), rng.binomial(1, 0.5, size=n), rng.binomial(1, 0.15, size=n)],
    )
    train_col = pd.Series(cats)
    test_col = pd.Series(["A", "B", "C"])

    train_encoded, test_encoded = _ordinal_encode_column(train_col, test_col, y, seed)

    rank_a = train_encoded[cats == "A"].mean()
    rank_b = train_encoded[cats == "B"].mean()
    rank_c = train_encoded[cats == "C"].mean()
    assert rank_a > rank_b > rank_c, f"expected A > B > C by rank (A highest positive rate), got A={rank_a}, B={rank_b}, C={rank_c}"
    assert test_encoded[0] > test_encoded[1] > test_encoded[2], "test encoding must preserve the same A > B > C ordering"
    print(f"PASS  test_ordinal_real_signal_is_captured (train ranks: A={rank_a:.2f}, B={rank_b:.2f}, C={rank_c:.2f})")


def test_ordinal_train_encoding_independent_of_test_data():
    n = 150
    seed = 3
    rng = np.random.RandomState(seed)
    cats = rng.choice(["x", "y", "z"], size=n)
    y = rng.randint(0, 2, size=n)
    train_col = pd.Series(cats)

    test_col_a = pd.Series(["x", "y"])
    test_col_b = pd.Series(["z", "z", "z", "z", "z", "unseen_category"])

    train_encoded_a, _ = _ordinal_encode_column(train_col, test_col_a, y, seed)
    train_encoded_b, _ = _ordinal_encode_column(train_col, test_col_b, y, seed)

    assert np.array_equal(train_encoded_a, train_encoded_b), (
        "train_encoded changed when only test_col changed — test data is leaking into the training encoding."
    )
    print("PASS  test_ordinal_train_encoding_independent_of_test_data")


def test_count_encoding_is_train_only():
    train_col = pd.Series(["a", "a", "a", "b", "b", "c"])
    test_col_small = pd.Series(["a"])
    test_col_huge = pd.Series(["a"] * 1000 + ["b"] * 1000)

    train_encoded_1, test_encoded_1 = _count_encode_column(train_col, test_col_small)
    train_encoded_2, test_encoded_2 = _count_encode_column(train_col, test_col_huge)

    assert np.array_equal(train_encoded_1, train_encoded_2), (
        "LEAK: train-side counts changed depending on the size/content of test_col — "
        "counts must come from the training column alone."
    )
    assert train_encoded_1[0] == 3.0
    assert test_encoded_1[0] == 3.0
    print("PASS  test_count_encoding_is_train_only")


def test_binary_encoding_is_train_only_and_reversible_cardinality():
    rng = np.random.RandomState(9)
    train_df = pd.DataFrame({"cat": rng.choice(["p", "q", "r", "s", "t"], 200)})
    test_df_a = pd.DataFrame({"cat": ["p", "q"]})
    test_df_b = pd.DataFrame({"cat": ["r"] * 500 + ["unseen"] * 500})

    train_enc_a, _ = _binary_encode_columns(train_df, test_df_a, ["cat"])
    train_enc_b, _ = _binary_encode_columns(train_df, test_df_b, ["cat"])

    bit_cols = [c for c in train_enc_a.columns if c.startswith("cat_bin")]
    assert len(bit_cols) == 3, f"expected ceil(log2(5))=3 bit columns, got {len(bit_cols)}"
    assert train_enc_a[bit_cols].equals(train_enc_b[bit_cols]), (
        "LEAK: train-side binary codes changed depending on test_df's content — "
        "the category->index mapping must come from training data alone."
    )
    print("PASS  test_binary_encoding_is_train_only_and_reversible_cardinality")


if __name__ == "__main__":
    print(f"(ORDINAL_ENCODING_FOLDS={ORDINAL_ENCODING_FOLDS})\n")
    test_ordinal_no_self_leakage_unique_categories()
    test_ordinal_real_signal_is_captured()
    test_ordinal_train_encoding_independent_of_test_data()
    test_count_encoding_is_train_only()
    test_binary_encoding_is_train_only_and_reversible_cardinality()
    print("\nALL NEW-ENCODING LEAKAGE TESTS PASSED")
