import numpy as np
import pandas as pd

from experiment_common import load_dataset, make_split
from pipeline import (
    _apply_cleaning_plan,
    _apply_derived_fills,
    _coerce_effectively_numeric,
    _discover_arithmetic_relationships,
    _discover_missingness_correlations,
    DEFAULT_CLEANING_PLAN,
)


def test_telco_discovery_finds_the_real_relationship():
    df, target = load_dataset("telco_churn")
    features = df.drop(columns=[target])
    numeric_view = _coerce_effectively_numeric(features)

    miss = _discover_missingness_correlations(numeric_view)
    arith = _discover_arithmetic_relationships(numeric_view)

    assert "TotalCharges" in miss, "missingness-correlation signal did not fire on Telco's TotalCharges at all"
    top = miss["TotalCharges"][0]
    assert top["column"] == "tenure" and top["value"] == 0.0, (
        f"expected tenure==0 to rank first (rarest, most distinguishing), got {top}"
    )
    assert top["concentration_in_missing_rows"] == 1.0

    assert "TotalCharges" in arith, "arithmetic-relationship signal did not fire on Telco's TotalCharges at all"
    rel = arith["TotalCharges"]
    assert rel["method"] == "product"
    assert set(rel["source_columns"]) == {"tenure", "MonthlyCharges"}
    assert rel["match_rate"] >= 0.90
    print(f"PASS  test_telco_discovery_finds_the_real_relationship (match_rate={rel['match_rate']})")


def test_telco_derived_fill_end_to_end():
    df, target = load_dataset("telco_churn")
    train_df, test_df = make_split(df, target, seed=42)

    plan = dict(DEFAULT_CLEANING_PLAN)
    plan["treat_as_null_columns"] = ["TotalCharges"]
    plan["coerce_numeric_columns"] = ["TotalCharges"]
    plan["derived_fills"] = {"TotalCharges": {"method": "product", "source_columns": ["tenure", "MonthlyCharges"]}}

    X_train, X_test, y_train, y_test, ok = _apply_cleaning_plan(train_df, test_df, target, "Yes", plan)
    assert ok

    zero_tenure_mask = train_df["tenure"] == 0
    assert zero_tenure_mask.sum() > 0, "test setup issue: no tenure==0 rows in this split"
    filled = X_train.loc[zero_tenure_mask, "TotalCharges"]
    assert (filled == 0.0).all(), f"expected TotalCharges==0 for all tenure==0 rows, got {filled.tolist()}"

    non_missing = train_df.loc[train_df["TotalCharges"].astype(str).str.strip() != "", "TotalCharges"].astype(float)
    assert non_missing.median() > 100, "test setup issue: unexpected TotalCharges scale"
    print("PASS  test_telco_derived_fill_end_to_end (all tenure==0 rows filled with 0.0, not the median)")


def test_no_spurious_relationship_on_synthetic_unrelated_data():
    rng = np.random.RandomState(0)
    n = 500
    df = pd.DataFrame({
        "a": rng.normal(100, 10, n),
        "b": rng.normal(50, 5, n),
        "c": rng.normal(20, 50, n),
        "d": rng.choice(["x", "y", "z"], n),
    })
    missing_idx = rng.choice(n, 40, replace=False)
    df.loc[missing_idx, "c"] = np.nan

    numeric_view = _coerce_effectively_numeric(df)
    miss = _discover_missingness_correlations(numeric_view)
    arith = _discover_arithmetic_relationships(numeric_view)

    assert "c" not in arith, f"SUSPICIOUS: found a spurious arithmetic relationship on unrelated synthetic data: {arith.get('c')}"
    if "c" in miss:
        assert miss["c"][0]["concentration_in_missing_rows"] < 0.95, (
            f"SUSPICIOUS: found near-total concentration on purely random missingness: {miss['c'][0]}"
        )
    print("PASS  test_no_spurious_relationship_on_synthetic_unrelated_data")


def test_derived_fill_is_row_local_not_a_cross_row_statistic():
    def make_df(seed_extra):
        rng = np.random.RandomState(seed_extra)
        n = 100
        tenure = rng.randint(1, 60, n)
        rate = rng.uniform(20, 100, n)
        total = tenure * rate
        df = pd.DataFrame({"tenure": tenure, "MonthlyCharges": rate, "TotalCharges": total, "y": rng.randint(0, 2, n)})
        df.loc[0:4, "tenure"] = 0
        df.loc[0:4, "TotalCharges"] = np.nan
        return df

    train_a = make_df(1)
    train_b = train_a.copy()
    train_b.loc[5:, "MonthlyCharges"] = train_b.loc[5:, "MonthlyCharges"] * 3 + 17

    test_df = pd.DataFrame({"tenure": [1], "MonthlyCharges": [10.0], "TotalCharges": [10.0], "y": [0]})
    rule = {"TotalCharges": {"method": "product", "source_columns": ["tenure", "MonthlyCharges"]}}

    filled_a, _ = _apply_derived_fills(train_a.copy(), test_df.copy(), "y", rule)
    filled_b, _ = _apply_derived_fills(train_b.copy(), test_df.copy(), "y", rule)

    assert np.array_equal(filled_a.loc[0:4, "TotalCharges"].to_numpy(), filled_b.loc[0:4, "TotalCharges"].to_numpy()), (
        "LEAK-SHAPED BUG: changing unrelated rows' values changed the derived fill for "
        "the missing rows — the fill is not purely row-local."
    )
    assert (filled_a.loc[0:4, "TotalCharges"] == 0.0).all()
    print("PASS  test_derived_fill_is_row_local_not_a_cross_row_statistic")


def test_discovery_finds_a_non_modal_category_relationship():
    n = 500
    rng = np.random.RandomState(2)
    smoking = rng.choice(
        ["never smoked", "formerly smoked", "smokes", "Unknown"], n, p=[0.4, 0.25, 0.2, 0.15],
    )
    assert pd.Series(smoking).mode().iloc[0] != "Unknown", "test setup issue: Unknown must NOT be the mode"
    bmi = rng.normal(28, 5, n)
    df = pd.DataFrame({"smoking_status": smoking, "bmi": bmi})
    unknown_mask = df["smoking_status"] == "Unknown"
    missing_idx = df.index[unknown_mask][: int(unknown_mask.sum() * 0.9)]
    df.loc[missing_idx, "bmi"] = np.nan

    found = _discover_missingness_correlations(df)
    assert "bmi" in found, "failed to discover a relationship concentrated at a NON-modal category"
    assert found["bmi"][0]["value"] == "Unknown"
    print("PASS  test_discovery_finds_a_non_modal_category_relationship")


def test_categorical_condition_zero_when():
    n = 60
    rng = np.random.RandomState(4)
    smoking = rng.choice(["never smoked", "formerly smoked", "smokes", "Unknown"], n)
    bmi = rng.normal(28, 5, n)
    df = pd.DataFrame({"smoking_status": smoking, "bmi": bmi, "y": rng.randint(0, 2, n)})
    unknown_mask = df["smoking_status"] == "Unknown"
    df.loc[unknown_mask, "bmi"] = np.nan
    assert unknown_mask.sum() > 0, "test setup issue: no Unknown rows"

    test_df = pd.DataFrame({"smoking_status": ["Unknown", "smokes"], "bmi": [np.nan, 27.0], "y": [0, 1]})
    rule = {"bmi": {"method": "zero_when", "when_column": "smoking_status", "when_value": "Unknown", "fill_value": 26.5}}

    filled_train, filled_test = _apply_derived_fills(df.copy(), test_df.copy(), "y", rule)

    assert (filled_train.loc[unknown_mask, "bmi"] == 26.5).all(), (
        f"expected all smoking_status=='Unknown' rows filled with 26.5, got {filled_train.loc[unknown_mask, 'bmi'].tolist()}"
    )
    assert filled_test.loc[0, "bmi"] == 26.5, "test row with smoking_status=='Unknown' should also be filled"
    assert filled_test.loc[1, "bmi"] == 27.0, "test row with a known bmi should be left untouched"
    print("PASS  test_categorical_condition_zero_when (categorical when_value now supported)")


def test_categorical_fill_target():
    n = 60
    rng = np.random.RandomState(6)
    work_type = rng.choice(["Private", "Self-employed", "Govt_job", "children"], n)
    smoking = rng.choice(["never smoked", "formerly smoked", "smokes"], n).astype(object)
    missing_mask = work_type == "children"
    smoking[missing_mask] = None
    df = pd.DataFrame({"work_type": work_type, "smoking_status": smoking, "y": rng.randint(0, 2, n)})
    assert missing_mask.sum() > 0, "test setup issue: no children rows"

    test_df = pd.DataFrame({"work_type": ["children", "Private"], "smoking_status": [None, "smokes"], "y": [0, 1]})
    rule = {
        "smoking_status": {
            "method": "zero_when", "when_column": "work_type", "when_value": "children",
            "fill_value": "never smoked",
        }
    }

    filled_train, filled_test = _apply_derived_fills(df.copy(), test_df.copy(), "y", rule)

    assert (filled_train.loc[missing_mask, "smoking_status"] == "never smoked").all(), (
        f"expected all work_type=='children' rows filled with 'never smoked', "
        f"got {filled_train.loc[missing_mask, 'smoking_status'].tolist()}"
    )
    assert filled_test.loc[0, "smoking_status"] == "never smoked", "test row with work_type=='children' should also be filled"
    assert filled_test.loc[1, "smoking_status"] == "smokes", "test row with a known smoking_status should be left untouched"
    print("PASS  test_categorical_fill_target (categorical fill_value now supported)")


def test_real_stroke_bmi_signal_is_actually_empty():
    from pipeline import MISSINGNESS_CONCENTRATION_THRESHOLD, MISSINGNESS_BASE_RATE_MARGIN
    df, target = load_dataset("stroke")
    for seed in (42, 123, 2024):
        train_df, _ = make_split(df, target, seed)
        features = train_df.drop(columns=[target])
        numeric_view = _coerce_effectively_numeric(features)
        miss = _discover_missingness_correlations(numeric_view)
        assert "bmi" not in miss, (
            f"seed={seed}: a real bmi/smoking_status relationship now exists ({miss.get('bmi')}) — "
            f"update this test's docstring, this is no longer a hallucination case."
        )
    print("PASS  test_real_stroke_bmi_signal_is_actually_empty (confirms the historical fallback was "
          "an ungrounded LLM claim, not a real discovery blocked by the schema)")


if __name__ == "__main__":
    test_telco_discovery_finds_the_real_relationship()
    test_telco_derived_fill_end_to_end()
    test_no_spurious_relationship_on_synthetic_unrelated_data()
    test_derived_fill_is_row_local_not_a_cross_row_statistic()
    test_discovery_finds_a_non_modal_category_relationship()
    test_categorical_condition_zero_when()
    test_categorical_fill_target()
    test_real_stroke_bmi_signal_is_actually_empty()
    print("\nALL CLEANING INVESTIGATION TESTS PASSED")
