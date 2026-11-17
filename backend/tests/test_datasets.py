from app.services.datasets import describe, preview
import pandas as pd


def test_dataset_summary_handles_missing_and_duplicates():
    frame = pd.DataFrame({"feature": [1, 1, None], "label": ["a", "a", "b"]})
    columns, missing, duplicates = describe(frame)
    assert columns["feature"] == "float64"
    assert missing["feature"] == 1
    assert duplicates == 1
    assert len(preview(frame)) == 3
