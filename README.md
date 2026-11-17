# Multi-output text classification

[![Tests](https://github.com/mukesh-mehta/multilabel-multiclass-classifcation/actions/workflows/ci.yml/badge.svg)](https://github.com/mukesh-mehta/multilabel-multiclass-classifcation/actions/workflows/ci.yml)

Predict several categorical attributes from the same text. The maintained CPU
baseline combines TF-IDF with one logistic-regression classifier per target,
using scikit-learn's
[MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html).
It trains, evaluates, saves an artifact, and predicts from the command line.

The repository name is retained for existing links. More precisely, this baseline
is **multi-output multiclass**: each target has exactly one category per example.
It does not yet predict arbitrary sets of labels within a single target.

## Quick start

Use Python 3.12. No GPU, downloaded language model, or private dataset is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python text_baseline.py train examples/demo.csv artifacts/demo --targets topic kind
python text_baseline.py predict artifacts/demo/model.joblib "How to debug Python code" "A football team won"
```

On Windows, activate with `.venv\Scripts\activate`. The example CSV is synthetic;
its scores only demonstrate that the workflow runs.

## Input and output contract

Supply UTF-8 CSV with a `text` column and any categorical target columns:

```csv
text,topic,kind
A football team won the final,sport,news
How to configure a database,tech,guide
```

Use more than these two example rows for training. Blank values, missing columns,
and duplicate text (after trimming and case-folding) are rejected. Every target
needs at least two classes in the training split. Class values are read as strings.

Training writes to a new or empty directory:

| File | Contents |
| --- | --- |
| `model.joblib` | Fitted text pipeline, target order, and artifact format version |
| `metrics.json` | Per-target accuracy/macro-F1, exact-match accuracy, unseen test labels |
| `split.json` | Zero-based CSV data-row positions used for training and testing |

The default split is 75% training and 25% evaluation, with seed 42. Change it with
`--test-size 0.2 --seed 7`. The TF-IDF vocabulary is fitted **only on training text**.
The random split is not stratified across target combinations; inspect the
reported unseen labels and use enough examples per category. Duplicate detection
does not identify near-duplicates, authors, threads, or time-based leakage.

Prediction returns a JSON list, e.g. `[{"topic":"tech","kind":"guide"}]`.
Only load artifacts you trust: joblib uses pickle and can execute code during
loading. Use the same dependency versions when training and loading artifacts.

## Features and roadmap

| Capability | Status |
| --- | --- |
| Configurable categorical target columns | Implemented |
| TF-IDF unigram/bigram features | Implemented |
| Independent logistic-regression heads | Implemented |
| Seeded holdout evaluation | Implemented |
| Per-target accuracy and macro-F1 | Implemented |
| Exact-match score and unseen-label reporting | Implemented |
| Duplicate/blank data validation | Implemented |
| Save/load and CLI prediction | Implemented |
| Group-aware and temporal splits | Planned |
| Multilabel sets within each target | Planned |
| Transformer baseline and model comparisons | Planned |
| Cross-validation and calibration | Planned |

## Historical experiment and validation

`assignment.ipynb` contains the original Keras experiment with 65 and 719 output
categories. Its original dependency list is preserved in
[docs/requirements-legacy.txt](docs/requirements-legacy.txt). That notebook is not
part of the modern quick start and has not been rerun against its original data.

CI exercises real fitting, prediction, deterministic splitting, artifact reload,
and validation failures on small synthetic data. It does not claim competitive
accuracy on the original task. Original notes: [docs/ORIGINAL_README.md](docs/ORIGINAL_README.md).

Maintained code uses [MIT](LICENSE). Historical notebooks and third-party material
are excluded as explained in [NOTICE.md](NOTICE.md). Contributions:
[CONTRIBUTING.md](CONTRIBUTING.md).
