import pandas as pd
from app.services.recommender import recommend_models


def test_recommend_models_classification():
    df = pd.DataFrame({
        "num1": list(range(20)),
        "cat1": ["a", "b"] * 10,
        "target": [0, 1] * 10
    })
    res = recommend_models(df, target_column="target", problem_type="classification")
    assert res["problem_type"] == "classification"
    assert len(res["recommended_algorithms"]) >= 3
    names = [r["algorithm"] for r in res["recommended_algorithms"]]
    assert "logistic_regression" in names
    assert "random_forest" in names
    assert "gradient_boosting" in names
    assert "decision_tree" in names
    assert "extra_trees" in names
    assert len(res["explanation"]) > 0


def test_recommend_models_regression():
    df = pd.DataFrame({
        "feature_x": [float(i) for i in range(30)],
        "target": [float(i * 2.5 + 1.0) for i in range(30)]
    })
    res = recommend_models(df, target_column="target", problem_type="regression")
    assert res["problem_type"] == "regression"
    names = [r["algorithm"] for r in res["recommended_algorithms"]]
    assert "linear_regression" in names
    assert "random_forest" in names
    assert "gradient_boosting" in names
