"""Train and evaluate a CPU baseline for multiple categorical text targets."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def read_data(path, targets):
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = ["text", *targets]
    if not targets or len(set(targets)) != len(targets) or "text" in targets:
        raise ValueError("Choose distinct target columns other than text")
    if not set(columns).issubset(data.columns) or data.empty:
        raise ValueError(f"CSV requires nonempty columns: {columns}")
    if any(data[col].str.strip().eq("").any() for col in columns):
        raise ValueError("Text and target values cannot be blank")
    normalized = data.text.str.strip().str.casefold()
    if normalized.duplicated().any():
        raise ValueError("Duplicate text must be resolved before splitting")
    return data


def train(csv, targets, output, *, seed=42, test_size=0.25):
    frame = read_data(csv, targets)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    train_rows, test_rows = train_test_split(np.arange(len(frame)), test_size=test_size, random_state=seed)
    for column in targets:
        if frame.iloc[train_rows][column].nunique() < 2:
            raise ValueError(f"Training split needs at least two classes for {column}; add examples or change split")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)),
        ("classifier", MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=seed))),
    ])
    pipeline.fit(frame.iloc[train_rows].text, frame.iloc[train_rows][targets])
    actual = frame.iloc[test_rows][targets].to_numpy()
    predicted = pipeline.predict(frame.iloc[test_rows].text)
    report = {"seed": seed, "train_rows": len(train_rows), "test_rows": len(test_rows),
              "exact_match": float(np.all(actual == predicted, axis=1).mean()), "targets": {}}
    for i, column in enumerate(targets):
        known = set(frame.iloc[train_rows][column])
        report["targets"][column] = {
            "accuracy": float(accuracy_score(actual[:, i], predicted[:, i])),
            "macro_f1": float(f1_score(actual[:, i], predicted[:, i], average="macro", zero_division=0)),
            "unseen_test_labels": sorted(set(actual[:, i]) - known),
        }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "targets": targets, "format_version": 1}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "split.json").write_text(json.dumps({"train": train_rows.tolist(), "test": test_rows.tolist()}, indent=2) + "\n")
    return report


def predict(artifact, texts):
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Provide at least one nonempty text")
    # joblib uses pickle: callers must only supply trusted artifacts.
    saved = joblib.load(artifact)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported artifact version")
    return [dict(zip(saved["targets"], row)) for row in saved["pipeline"].predict(texts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("csv", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--targets", nargs="+", required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--test-size", type=float, default=0.25)
    inference = commands.add_parser("predict")
    inference.add_argument("artifact", type=Path)
    inference.add_argument("text", nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.csv, args.targets, args.output, seed=args.seed, test_size=args.test_size)
        else:
            result = predict(args.artifact, args.text)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
