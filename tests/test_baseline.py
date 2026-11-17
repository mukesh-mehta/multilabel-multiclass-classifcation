import json
import pandas as pd
import pytest
from text_baseline import predict, read_data, train


def sample(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame([{"text": f"{'football team' if i % 2 else 'software code'} example {i}",
                   "topic": "sport" if i % 2 else "tech", "priority": "high" if i % 3 else "low"}
                  for i in range(40)]).to_csv(path, index=False)
    return path


def test_train_save_reload_and_split(tmp_path):
    csv = sample(tmp_path)
    result = train(csv, ["topic", "priority"], tmp_path / "model")
    assert result["train_rows"] == 30
    assert set(result["targets"]) == {"topic", "priority"}
    rows = predict(tmp_path / "model/model.joblib", ["software code", "football team"])
    assert rows[0]["topic"] == "tech"
    assert rows[1]["topic"] == "sport"
    split = json.loads((tmp_path / "model/split.json").read_text())
    assert not set(split["train"]) & set(split["test"])
    assert len(set(split["train"] + split["test"])) == 40
    assert train(csv, ["topic", "priority"], tmp_path / "repeat") == result
    with pytest.raises(ValueError, match="empty"):
        train(csv, ["topic", "priority"], tmp_path / "model")


def test_validation(tmp_path):
    csv = sample(tmp_path)
    for targets in [[], ["missing"], ["topic", "topic"], ["text"]]:
        with pytest.raises(ValueError):
            read_data(csv, targets)
    frame = pd.read_csv(csv)
    frame.loc[1, "text"] = frame.loc[0, "text"].upper()
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_data(csv, ["topic"])


def test_blank_predictions_rejected():
    with pytest.raises(ValueError):
        predict("not-opened.joblib", [" "])
