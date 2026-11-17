import pandas as pd
from app.services.profiler import profile_dataset


def test_profiler_basic():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 100],  # 100 is an outlier
        "city": ["NY", "SF", "NY", "LA", "NY"],
        "constant_col": [1, 1, 1, 1, 1],
        "target": [0, 0, 1, 1, 1]
    })
    res = profile_dataset(df, target_column="target")
    assert res["row_count"] == 5
    assert res["column_count"] == 4
    assert "age" in res["numerical_columns"]
    assert "city" in res["categorical_columns"]
    assert res["quality_score"] < 100.0  # constant column deduction
    warnings = [w["message"] for w in res["warnings"]]
    assert any("constant" in w.lower() for w in warnings)
    assert res["target_analysis"] is not None
    assert res["target_analysis"]["target_column"] == "target"
    assert len(res["suggested_operations"]) > 0


def test_profiler_missing_and_imbalance():
    df = pd.DataFrame({
        "feat1": [None, None, None, 1.0, 2.0],  # 60% missing
        "feat2": ["a", "b", "c", "d", "e"],
        "target": ["rare", "common", "common", "common", "common"]  # 4:1 imbalance
    })
    res = profile_dataset(df, target_column="target")
    assert any("missing values" in w["message"].lower() for w in res["warnings"])
    assert any("imbalanced" in w["message"].lower() for w in res["warnings"])
    assert res["target_analysis"]["is_imbalanced"] is True
