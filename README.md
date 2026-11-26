# ML Experiment Reviewer (`mlreview`)

Finds the bugs that **silently ruin machine-learning results**: data leakage, train/test
contamination, misleading metrics, missing seeds and common PyTorch mistakes. Point it at a
training script (`.py`) or notebook (`.ipynb`) and get a report ranked by severity, with a
concrete fix for every finding.

- **Two layers:** 19 deterministic rules (fast, repeatable, offline) plus an optional AI second
  pass (Groq) that doubts false alarms and finds issues rules cannot see.
- **Safe:** your code is parsed, never executed. The rules need nothing but Python 3.9+.
- **Practical:** Markdown / JSON / text output, folder scanning, `--fail-on` for CI, and
  per-line suppression comments.

## What it catches

| Rule | Severity | Problem |
|---|---|---|
| ML001 | critical | Scaler / imputer / encoder / PCA / feature selection fitted before the train/test split |
| ML002 | critical | `fit` / `fit_transform` called on test or validation data |
| ML003 | critical | Target column left inside the features |
| ML004 | critical | SMOTE / over- or under-sampling before the split |
| ML005 | critical | Model trained on the full dataset, then scored on the test set |
| ML006 | critical / medium | Preprocessing fitted outside cross-validation |
| ML007 | high | Target (mean) encoding computed on rows later used for evaluation |
| ML008 | high | Test set used for early stopping or hyper-parameter search |
| ML009 | high | Only training-set performance is reported |
| ML010 | medium | Preprocessing fitted on combined train + test (Kaggle pattern) |
| ML011 | medium | Missing values filled with statistics from all rows |
| ML012 | medium / high | Accuracy is the only metric for a classification task |
| ML013 | medium | ROC AUC / log loss computed from hard 0/1 predictions |
| ML014 | medium | Random split on time-ordered data |
| ML015 | medium | Random split while several rows share an entity id |
| ML016 | high | PyTorch: `optimizer.zero_grad()` missing |
| ML017 | medium | PyTorch: `model.eval()` never called with Dropout/BatchNorm |
| ML018 | low | Random seeds not set |
| ML019 | low | Classification split without `stratify` |

## Quick start (Windows, macOS, Linux)

Open a terminal (Git Bash, PowerShell or Terminal) in this folder.

```bash
# 1. Run the test suite (54 tests, no packages needed)
python -m unittest discover -s tests

# 2. Review the demo files
python mlreview.py examples/leaky_churn.py
python mlreview.py examples/leaky_kaggle.ipynb
python mlreview.py examples/torch_training.py
python mlreview.py examples/clean_churn.py          # -> PASS, no issues

# 3. See why it matters: run the leaky and the clean model (needs scikit-learn + pandas)
python -m venv .venv
# Windows:      .venv\Scripts\activate          macOS/Linux:  source .venv/bin/activate
pip install -r requirements-examples.txt
python examples/leaky_churn.py     # perfect 1.000 scores ... because of leakage
python examples/clean_churn.py     # the real performance (ROC AUC ~0.83, churn recall ~0.5)
```

On macOS/Linux use `python3` if `python` is not found.

## Review your own code

```bash
python mlreview.py path/to/train.py
python mlreview.py path/to/notebook.ipynb --out review.md     # save the Markdown report
python mlreview.py my_project/                                 # every .py / .ipynb in a folder
python mlreview.py train.py --format json                      # machine-readable
python mlreview.py train.py --min-severity high                # hide medium/low notes
python mlreview.py src/ --fail-on high                         # exit code 1 if high+ found (CI)
python mlreview.py --list-rules
```

Silence a finding you have checked, on that line:

```python
X = scaler.fit_transform(X)   # mlreview: ignore
y_pred = clf.predict(X_test)  # noqa: ML013
```

## Optional: AI second pass (Groq)

1. Get a free API key at https://console.groq.com/keys
2. Copy `.env.example` to `.env` and paste your key after `GROQ_API_KEY=`
   (never commit `.env`; it is in `.gitignore`).
3. Run:

```bash
python mlreview_agent.py examples/leaky_churn.py
python mlreview_agent.py path/to/notebook.ipynb --out review.md
python mlreview_agent.py train.py --model openai/gpt-oss-20b    # faster / cheaper model
```

The agent runs the rules first, then sends the code and the findings to Groq. The AI may:
- **doubt** a rule finding. It is not deleted; it moves to a "doubted by the AI pass" section
  so you can double-check it;
- **add** findings the rules cannot see, marked `AI` with confidence "AI suggestion - verify".

Every AI answer is validated (schema, severity, line numbers). If there is no key, the key is
wrong, or the API is down, you still get the full rules report with a note.

> Privacy: the AI pass sends your code to Groq. Use `mlreview.py` alone (fully offline) for
> code you cannot share.

## Files

| File | Purpose |
|---|---|
| `mlreview.py` | The reviewer: rules, notebook loader, reports, CLI. Standard library only |
| `mlreview_agent.py` | Optional AI second pass using Groq |
| `examples/leaky_churn.py` | Runnable model with 6 hidden bugs (scores a fake-perfect 1.000) |
| `examples/clean_churn.py` | Same model done correctly (reviewer reports nothing) |
| `examples/leaky_kaggle.ipynb` | Kaggle-style notebook with 9 issues (static review only) |
| `examples/torch_training.py` | PyTorch loop with 3 issues (static review only) |
| `tests/` | 54 tests: every rule, false-positive guards, notebooks, CLI, AI pass with a fake API |

## Limits (what static rules cannot see)

- Duplicate rows shared by train and test, or leakage created in SQL / other files.
- Features that only exist after the outcome (the AI pass can catch some of these by meaning).
- Leakage through complex index juggling the rules cannot follow.

Verified on scikit-learn's 282 official examples: no crashes, and every critical/high
finding was a real leak or a training-only evaluation.

## License

MIT (c) 2026 Muhammad Waqas
