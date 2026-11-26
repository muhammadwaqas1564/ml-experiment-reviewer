#!/usr/bin/env python3
"""mlreview - ML Experiment Reviewer.

Reads a training script (.py) or notebook (.ipynb) and finds the bugs that
silently ruin ML results: data leakage, train/test contamination, misleading
metrics, missing seeds and common PyTorch training mistakes. It returns a
report ranked by severity, with a fix for every finding.

The code under review is parsed, never executed. Standard library only,
Python 3.9+.

Usage:
    python mlreview.py train.py
    python mlreview.py notebook.ipynb --format json --out review.json
    python mlreview.py src/ --fail-on high          # use in CI
    python mlreview.py --list-rules

Silence a finding on one line with a comment:  # mlreview: ignore
or for one rule only:                           # noqa: ML012
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

__version__ = "1.0.0"

SEVERITIES = ("critical", "high", "medium", "low")
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}

# ----------------------------------------------------------------- rule catalogue
RULES = {
    "ML001": ("critical", "Preprocessing fitted before the train/test split"),
    "ML002": ("critical", "Model or preprocessing fitted on test/validation data"),
    "ML003": ("critical", "Target column left inside the features"),
    "ML004": ("critical", "Resampling (SMOTE / over- or under-sampling) before the split"),
    "ML005": ("critical", "Model trained on the full dataset, then scored on the test set"),
    "ML006": ("high", "Preprocessing fitted outside cross-validation"),
    "ML007": ("high", "Target encoding computed on rows that are later used for evaluation"),
    "ML008": ("high", "Test set used for early stopping or model selection"),
    "ML009": ("high", "Only training-set performance is reported"),
    "ML010": ("medium", "Preprocessing fitted on combined train + test data"),
    "ML011": ("medium", "Missing values filled with statistics from all rows"),
    "ML012": ("medium", "Accuracy is the only metric for a classification task"),
    "ML013": ("medium", "ROC AUC / log loss computed from hard class predictions"),
    "ML014": ("medium", "Random split on time-ordered data"),
    "ML015": ("medium", "Random split while rows share an entity id"),
    "ML016": ("high", "PyTorch: gradients never reset (zero_grad missing)"),
    "ML017": ("medium", "PyTorch: model.eval() never called"),
    "ML018": ("low", "Random seeds not set (results not reproducible)"),
    "ML019": ("low", "Classification split without stratify"),
}

TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer",
    "QuantileTransformer", "PowerTransformer", "SimpleImputer", "KNNImputer",
    "IterativeImputer", "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "PCA",
    "TruncatedSVD", "KernelPCA", "FastICA", "NMF", "SelectKBest", "SelectPercentile",
    "SelectFromModel", "RFE", "RFECV", "VarianceThreshold", "GenericUnivariateSelect",
    "TfidfVectorizer", "CountVectorizer", "TfidfTransformer", "KBinsDiscretizer",
    "ColumnTransformer", "make_column_transformer", "LabelEncoder", "LabelBinarizer",
    "MultiLabelBinarizer", "Binarizer", "SplineTransformer", "LinearDiscriminantAnalysis",
}
RESAMPLERS = {
    "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "SMOTENC", "SMOTEN",
    "RandomOverSampler", "RandomUnderSampler", "NearMiss", "TomekLinks",
    "EditedNearestNeighbours", "SMOTETomek", "SMOTEENN", "ClusterCentroids",
}
SEARCHERS = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV",
             "HalvingRandomSearchCV", "BayesSearchCV"}
PIPELINES = {"Pipeline", "make_pipeline", "ImbPipeline"}
CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict", "learning_curve",
            "validation_curve", "permutation_test_score"}
RANDOM_SPLITTERS = {"KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit",
                    "RepeatedKFold", "RepeatedStratifiedKFold"}
GROUP_SPLITTERS = {"GroupKFold", "GroupShuffleSplit", "StratifiedGroupKFold",
                   "LeaveOneGroupOut", "LeavePGroupsOut"}
CLASS_METRICS_OTHER = {
    "f1_score", "fbeta_score", "precision_score", "recall_score", "roc_auc_score",
    "average_precision_score", "balanced_accuracy_score", "classification_report",
    "confusion_matrix", "matthews_corrcoef", "cohen_kappa_score", "log_loss",
    "precision_recall_fscore_support", "precision_recall_curve", "roc_curve",
    "ConfusionMatrixDisplay", "RocCurveDisplay", "PrecisionRecallDisplay", "brier_score_loss",
}
METRICS = CLASS_METRICS_OTHER | {
    "accuracy_score", "r2_score", "mean_squared_error", "mean_absolute_error",
    "root_mean_squared_error", "mean_absolute_percentage_error", "explained_variance_score",
    "median_absolute_error", "mean_squared_log_error",
}
PROBA_METRICS = {"roc_auc_score", "average_precision_score", "log_loss", "brier_score_loss",
                 "roc_curve", "precision_recall_curve"}
FIT_METHODS = {"fit", "fit_transform", "fit_resample", "partial_fit", "fit_predict"}
EVAL_METHODS = {"predict", "predict_proba", "decision_function", "score", "evaluate"}

MODEL_RE = re.compile(r"(Classifier|Regressor|Regression|SVC|SVR|NB|Ridge|Lasso|ElasticNet|"
                      r"Sequential|Model|Booster|Network|Net|KNN)$")
CLASSIFIER_RE = re.compile(r"(Classifier|LogisticRegression|SVC|NB|LinearDiscriminantAnalysis)$")
RANDOM_EST_RE = re.compile(r"^(RandomForest|ExtraTrees?|GradientBoosting|HistGradientBoosting|"
                           r"DecisionTree|XGB|LGBM|CatBoost|MLP|SGD|KMeans|MiniBatchKMeans|"
                           r"Bagging|AdaBoost|IsolationForest|TSNE|Perceptron|RBFSampler|Nystroem|SMOTE|ADASYN|"
                           r"BorderlineSMOTE|SVMSMOTE|RandomOverSampler|RandomUnderSampler)")
TEST_RE = re.compile(r"test|valid|holdout|(^|_)val($|_|\d)|(^|_)dev($|_)", re.I)
STRICT_TEST_RE = re.compile(r"test|holdout", re.I)
TRAIN_RE = re.compile(r"train|(^|_)tr($|_|\d)", re.I)
Y_RE = re.compile(r"^(y|Y|y_?(all|full|data|raw)|target|targets|labels?)$")
X_RE = re.compile(r"^(X|x|X_?(all|full|data|raw|df)|features|feats|data_x)$")
ID_RE = re.compile(r"^(patient|user|customer|subject|session|player|account|device|client|"
                   r"person|member|store|household|company)_?id$", re.I)
IMBALANCE_RE = re.compile(r"value_counts|class_weight|smote|imbalanc|fraud|churn|anomal|"
                          r"minority|weights\s*=\s*\[", re.I)
IGNORE_RE = re.compile(r"mlreview:\s*ignore", re.I)


# ------------------------------------------------------------------- data classes
@dataclass
class Finding:
    rule: str
    severity: str
    title: str
    line: int
    why: str
    fix: str
    confidence: str = "high"
    code: str = ""
    location: str = ""
    source: str = "rules"          # "rules" or "ai"
    note: str = ""                 # e.g. why the AI pass doubted it


@dataclass
class Review:
    file: str
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    dismissed: list = field(default_factory=list)
    ai_summary: str = ""
    lines_of_code: int = 0
    error: str = ""

    def counts(self) -> dict:
        c = {s: 0 for s in SEVERITIES}
        for f in self.findings:
            c[f.severity] += 1
        return c

    def verdict(self) -> str:
        if self.error:
            return "NOT REVIEWED"
        c = self.counts()
        if c["critical"]:
            return "FAIL - results are not trustworthy until the critical issues are fixed"
        if c["high"]:
            return "WARNING - results are likely biased; fix the high-severity issues"
        if c["medium"] or c["low"]:
            return "PASS WITH NOTES - no leakage found by the rules; see the notes below"
        return "PASS - no issues found by the rules"


# ------------------------------------------------------------------- AST helpers
def dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted(node.func) + "()"
    if isinstance(node, ast.Subscript):
        return dotted(node.value) + "[]"
    return ""


def last(name: str) -> str:
    return name.replace("()", "").replace("[]", "").rsplit(".", 1)[-1]


def base_name(node):
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Subscript, ast.Attribute, ast.Starred)):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        else:
            return None


def names_in(node) -> set:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def strings_in(node) -> set:
    if node is None:
        return set()
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def target_names(t) -> set:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        out = set()
        for e in t.elts:
            out |= target_names(e)
        return out
    if isinstance(t, ast.Starred):
        return target_names(t.value)
    b = base_name(t)
    return {b} if b else set()


def method(call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else ""


def func_name(call) -> str:
    return last(dotted(call.func))


def kw(call, name):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def has_kwargs_splat(call) -> bool:
    return any(k.arg is None for k in call.keywords)


def is_false(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


# ------------------------------------------------------------------ source loading
def load_source(path: Path):
    """Return (code, linemap, notes). linemap[i] = (cell_no, line_in_cell) or None."""
    notes = []
    if path.suffix.lower() != ".ipynb":
        return path.read_text(encoding="utf-8", errors="replace"), None, notes
    nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    cells = nb.get("cells")
    if cells is None:  # very old nbformat
        cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    lines, linemap = [], []
    for ci, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        cell_lines = src.splitlines()
        cleaned = []
        for ln in cell_lines:
            s = ln.lstrip()
            # IPython magics / shell escapes are not Python: keep the line count, comment them out
            if s.startswith(("%", "!", "?")) or s.endswith("?") and not s.startswith("#") and "=" not in s:
                cleaned.append("# " + ln)
            else:
                cleaned.append(ln)
        try:
            ast.parse("\n".join(cleaned))
        except SyntaxError as exc:
            notes.append(f"cell {ci} skipped: it does not parse as Python ({exc.msg}, line {exc.lineno})")
            cleaned = ["# " + ln for ln in cell_lines]
        for i, ln in enumerate(cleaned, start=1):
            lines.append(ln)
            linemap.append((ci, i))
        lines.append("")
        linemap.append((ci, len(cleaned) + 1))
    return "\n".join(lines), linemap, notes


# ----------------------------------------------------------------------- analyzer
class Analyzer:
    def __init__(self, code: str, linemap=None):
        self.code = code
        self.src_lines = code.splitlines()
        self.linemap = linemap
        self.tree = ast.parse(code)
        self.findings: list[Finding] = []
        self._seen = set()
        self._collect()

    # ---- collection
    def _collect(self):
        self.calls = sorted((n for n in ast.walk(self.tree) if isinstance(n, ast.Call)),
                            key=lambda n: (n.lineno, n.col_offset))
        self.imports = set()
        self.assigns = []        # (line, targets, value_names, node, value)
        self.var_types = {}
        self.dels = []           # (line, base, strings)
        self.full_targets = {}   # id(assign node) -> names that are fully reassigned
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Import):
                self.imports |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                self.imports.add(n.module.split(".")[0])
            elif isinstance(n, ast.Delete):
                for t in n.targets:
                    self.dels.append((n.lineno, base_name(t), strings_in(t)))
            if isinstance(n, ast.Assign):
                tg, value, raw_targets = set(), n.value, n.targets
                for t in n.targets:
                    tg |= target_names(t)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                tg, value, raw_targets = target_names(n.target), n.value, [n.target]
            else:
                continue
            full = set()
            for t in raw_targets:
                if isinstance(t, ast.Name):
                    full.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    full |= {e.id for e in t.elts if isinstance(e, ast.Name)}
            self.assigns.append((n.lineno, tg, names_in(value), n, value))
            self.full_targets[id(n)] = full
            if isinstance(value, ast.Call) and len(raw_targets) == 1 and \
                    isinstance(raw_targets[0], (ast.Name, ast.Attribute)):
                cls = self._constructed_class(value)
                if cls:
                    self.var_types[dotted(raw_targets[0])] = cls
        # in-place mutations (df.fillna(..., inplace=True)) behave like assignments
        for c in self.calls:
            v = kw(c, "inplace")
            if isinstance(v, ast.Constant) and v.value is True:
                b = base_name(c.func)
                if b:
                    self.assigns.append((c.lineno, {b}, names_in(c), c, c))
        self.assigns.sort(key=lambda a: a[0])
        self.call_targets = {}
        for ln, tg, _vn, _node, value in self.assigns:
            for sub in ast.walk(value):
                if isinstance(sub, ast.Call):
                    self.call_targets.setdefault(id(sub), tg)
        self.strings = strings_in(self.tree)
        self.all_names = {dotted(n) for n in ast.walk(self.tree) if isinstance(n, (ast.Name, ast.Attribute))}
        self.attr_names = {n.attr for n in ast.walk(self.tree) if isinstance(n, ast.Attribute)} | \
            {n.id for n in ast.walk(self.tree) if isinstance(n, ast.Name)}
        self.func_names = {func_name(c) for c in self.calls}
        self.methods = {method(c) for c in self.calls}
        self._collect_splits()
        self._collect_targets()

    def _constructed_class(self, value):
        """Class name for `x = Cls(...)` or `x = Cls(...).fit(...)`."""
        name = func_name(value)
        if isinstance(value.func, ast.Attribute) and value.func.attr in FIT_METHODS \
                and isinstance(value.func.value, ast.Call):
            name = func_name(value.func.value)
        if name and (name[0].isupper() or name in PIPELINES or name == "make_column_transformer"):
            return name
        return None

    def _collect_splits(self):
        self.splits = []          # (line, {source names}, {output names}, {feature source names})
        for c in self.calls:
            if func_name(c) == "train_test_split":
                srcs = {base_name(a) for a in c.args} - {None}
                for k in ("X", "arrays"):
                    if kw(c, k) is not None:
                        srcs.add(base_name(kw(c, k)))
                srcs.discard(None)
                xs = {base_name(c.args[0])} if c.args else {base_name(kw(c, "X"))}
                if srcs:
                    self.splits.append((c.lineno, srcs, self.call_targets.get(id(c), set()), xs - {None} or srcs))
        for ln, tg, _vn, _node, value in self.assigns:
            if not any(TRAIN_RE.search(t) or STRICT_TEST_RE.search(t) for t in tg):
                continue
            manual = isinstance(value, ast.Subscript) or (
                isinstance(value, ast.Call) and method(value) in {"sample", "head", "tail"})
            b = base_name(value)
            if manual and b and not TRAIN_RE.search(b) and not TEST_RE.search(b):
                self.splits.append((ln, {b}, set(tg), {b}))
        self.splits.sort(key=lambda s: s[0])
        self.split_sources = set().union(*(s[1] for s in self.splits)) if self.splits else set()
        self.first_split = self.splits[0][0] if self.splits else None

    def _collect_targets(self):
        self.targets = []         # (line, dataframe name, target column)
        for ln, tg, _vn, _node, v in self.assigns:
            if len(tg) != 1 or not Y_RE.match(next(iter(tg))):
                continue
            if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name):
                s = strings_in(v.slice)
                if len(s) == 1:
                    self.targets.append((ln, v.value.id, s.pop()))
            elif isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) \
                    and v.attr not in {"values", "target", "data", "shape", "index", "columns", "T"}:
                self.targets.append((ln, v.value.id, v.attr))
        self.target_cols = {t[2] for t in self.targets}

    # ---- helpers
    def receiver_class(self, call):
        f = call.func
        if not isinstance(f, ast.Attribute):
            return None
        v = f.value
        if isinstance(v, ast.Call):
            return func_name(v)
        if isinstance(v, (ast.Name, ast.Attribute)):
            return self.var_types.get(dotted(v))
        return None

    @staticmethod
    def kind(cls):
        if not cls:
            return "unknown"
        if cls in RESAMPLERS:
            return "resampler"
        if cls in TRANSFORMERS:
            return "transformer"
        if cls in SEARCHERS:
            return "searcher"
        if cls in PIPELINES:
            return "pipeline"
        if MODEL_RE.search(cls):
            return "model"
        return "unknown"

    def taint_from(self, names, start, until=None) -> set:
        """Names whose value derives from `names` between lines start..until."""
        t = {n for n in names if n}
        for ln, tg, vn, node, _v in self.assigns:
            if ln < start:
                continue
            if until is not None and ln > until:
                break
            if vn & t:
                t |= tg
            elif ln > start:
                t -= self.full_targets.get(id(node), set())   # overwritten with clean data
        return t

    def reassigned(self, name, after, before) -> bool:
        return any(after < ln < before and name in self.full_targets.get(id(node), set())
                   for ln, _tg, _vn, node, _v in self.assigns)

    def fit_outputs(self, call) -> set:
        """Variables that carry what a fit learned: assigned results plus the fitted object."""
        out = set(self.call_targets.get(id(call), set()))
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, (ast.Name, ast.Attribute)):
            out.add(dotted(f.value))
        return out

    def applied_to_split(self, receivers, after_line, split_targets, fit_line=None) -> bool:
        """Is a fitted object later used (transform/predict...) on data produced by the split?"""
        for c in self.calls:
            if c.lineno <= after_line or not isinstance(c.func, ast.Attribute):
                continue
            r = dotted(c.func.value)
            if fit_line is not None and self.reassigned(r, fit_line, c.lineno):
                continue   # the name now holds a different object
            if r in receivers and c.func.attr in {
                    "transform", "predict", "predict_proba", "score", "decision_function", "inverse_transform"}:
                if set().union(*(names_in(a) for a in c.args)) & split_targets if c.args else False:
                    return True
        return False

    def text(self, line) -> str:
        return self.src_lines[line - 1].strip() if 0 < line <= len(self.src_lines) else ""

    def location(self, line) -> str:
        if self.linemap and 0 < line <= len(self.linemap):
            cell, cl = self.linemap[line - 1]
            return f"cell {cell}, line {cl}"
        return f"line {line}"

    def suppressed(self, line, rule) -> bool:
        t = self.src_lines[line - 1] if 0 < line <= len(self.src_lines) else ""
        if IGNORE_RE.search(t):
            return True
        m = re.search(r"noqa\s*:\s*([\w, ]+)", t, re.I)
        return bool(m and rule in {x.strip().upper() for x in m.group(1).split(",")})

    def add(self, rule, line, why, fix, severity=None, confidence="high", title=None):
        if (rule, line) in self._seen or self.suppressed(line, rule):
            return
        self._seen.add((rule, line))
        sev, default_title = RULES[rule]
        self.findings.append(Finding(rule=rule, severity=severity or sev, title=title or default_title,
                                     line=line, why=why, fix=fix, confidence=confidence,
                                     code=self.text(line), location=self.location(line)))

    def preprocessing_fits(self):
        """Yield (call, kind, cls, arg) for transformer/resampler fits."""
        for c in self.calls:
            m = method(c)
            if m not in {"fit", "fit_transform", "fit_resample"} or not c.args:
                continue
            cls = self.receiver_class(c)
            k = self.kind(cls)
            if m == "fit_resample":
                k = "resampler" if k in {"unknown", "resampler"} else k
            elif m == "fit_transform" and k == "unknown":
                k = "transformer"
            if k in {"transformer", "resampler"}:
                yield c, k, cls or ("resampler" if k == "resampler" else "transformer"), c.args[0]

    def is_classification(self) -> bool:
        if any(CLASSIFIER_RE.search(t or "") for t in self.var_types.values()):
            return True
        if any(CLASSIFIER_RE.search(n) for n in self.func_names):
            return True
        return bool(self.func_names & (CLASS_METRICS_OTHER | {"accuracy_score"}))

    # ------------------------------------------------------------------ rules
    def run(self):
        for rule in (self.r_fit_before_split, self.r_fit_on_test, self.r_target_in_features,
                     self.r_full_fit_then_test, self.r_outside_cv, self.r_target_encoding,
                     self.r_test_for_selection, self.r_train_only_metrics, self.r_combined_fit,
                     self.r_fillna_stats, self.r_accuracy_only, self.r_proba_metrics,
                     self.r_time_split, self.r_group_split, self.r_torch, self.r_seeds,
                     self.r_stratify):
            rule()
        self.findings.sort(key=lambda f: (SEV_RANK[f.severity], f.line, f.rule))
        return self.findings

    def r_fit_before_split(self):
        self.leaky_fit_lines = set()
        for c, k, cls, arg in self.preprocessing_fits():
            b = base_name(arg)
            outs = self.fit_outputs(c)
            for split_line, srcs, split_out, xsrcs in self.splits:
                srcs = xsrcs   # leakage through preprocessing matters on the feature side
                if c.lineno < split_line:
                    fitted_on_split_data = bool(b and self.taint_from({b}, c.lineno, split_line) & srcs)
                    hit = fitted_on_split_data and (
                        bool(self.taint_from(outs, c.lineno, split_line) & srcs)
                        or self.applied_to_split(outs, split_line, split_out, c.lineno))
                else:
                    hit = bool(b and b in srcs and not TRAIN_RE.search(b) and not TEST_RE.search(b)
                               and not self.reassigned(b, split_line, c.lineno)
                               and self.applied_to_split(outs, c.lineno, split_out, c.lineno))
                if not hit:
                    continue
                self.leaky_fit_lines.add(c.lineno)
                if k == "resampler":
                    self.add("ML004", c.lineno,
                             why=(f"{cls} creates or removes rows using all of `{b}` before it is split. "
                                  "Synthetic rows built from test samples end up in training, and the test "
                                  "set no longer has the real class balance, so scores are inflated."),
                             fix=("Split first, then resample only the training set, or put the sampler "
                                  "inside an imblearn Pipeline so it runs per CV fold:\n"
                                  "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)\n"
                                  "X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)"))
                else:
                    self.add("ML001", c.lineno,
                             why=(f"{cls} learns statistics (mean, scale, vocabulary, selected features...) "
                                  f"from all rows of `{b}`, including rows that later become the test set. "
                                  "Information from the test set leaks into training and the test score "
                                  "is optimistic."),
                             fix=("Split first, then fit on the training rows only, or use a Pipeline:\n"
                                  "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n"
                                  f"pipe = make_pipeline({cls if cls[0].isupper() else 'StandardScaler'}(), model)\n"
                                  "pipe.fit(X_train, y_train)\n"
                                  "pipe.score(X_test, y_test)"))
                break

    def r_fit_on_test(self):
        for c in self.calls:
            m = method(c)
            if m not in FIT_METHODS or not c.args:
                continue
            b = base_name(c.args[0])
            if not b or not TEST_RE.search(b) or TRAIN_RE.search(b):
                continue
            cls = self.receiver_class(c)
            k = self.kind(cls)
            if m == "fit_transform" and k == "unknown":
                k = "transformer"
            what = "a model" if k in {"model", "pipeline", "searcher"} else (
                "a preprocessing step" if k in {"transformer", "resampler"} else "an estimator")
            self.add("ML002", c.lineno,
                     why=(f"`{m}` is called on `{b}`, so {what} learns from held-out data. "
                          "The evaluation set must only ever be transformed or predicted on; "
                          "fitting on it makes the reported score meaningless."
                          + (" Re-fitting a transformer on the test set also maps values differently "
                             "from training." if k == "transformer" else "")),
                     fix=(f"Fit on the training data and only transform/predict on `{b}`:\n"
                          "scaler.fit(X_train)\n"
                          f"{b}_scaled = scaler.transform({b})   # never fit / fit_transform here"))

    def r_target_in_features(self):
        for ln, tg, _vn, _node, v in self.assigns:
            if len(tg) != 1 or not X_RE.match(next(iter(tg))):
                continue
            xname = next(iter(tg))
            for _tl, df, col in self.targets:
                if base_name(v) != df or col in strings_in(v):
                    continue
                leaky = isinstance(v, ast.Name) or (isinstance(v, ast.Attribute) and v.attr == "values") or (
                    isinstance(v, ast.Call) and method(v) in {"drop", "copy", "to_numpy", "astype",
                                                              "select_dtypes", "fillna", "dropna"})
                if not leaky or self._target_removed(df, col, ln):
                    continue
                self.add("ML003", ln,
                         why=(f"`{xname}` is built from `{df}`, but the target column '{col}' is never "
                              "removed from it. The model can read the answer directly, which usually "
                              "gives near-perfect scores that collapse in production."),
                         fix=(f"Drop the target when building the features:\n"
                              f"y = {df}['{col}']\n"
                              f"{xname} = {df}.drop(columns=['{col}'])"),
                         confidence="high" if isinstance(v, ast.Name) else "medium")

    def _target_removed(self, df, col, before_line) -> bool:
        for c in self.calls:
            if c.lineno >= before_line:
                break
            if method(c) in {"drop", "pop"} and base_name(c.func) == df and col in strings_in(c):
                return True
        return any(ln < before_line and b == df and col in s for ln, b, s in self.dels)

    def r_full_fit_then_test(self):
        if not self.splits:
            return
        for c in self.calls:
            if method(c) != "fit" or not c.args:
                continue
            k = self.kind(self.receiver_class(c))
            if k not in {"model", "pipeline", "searcher"} and not (k == "unknown" and len(c.args) >= 2):
                continue
            b = base_name(c.args[0])
            if not b or b not in self.split_sources or TRAIN_RE.search(b) or TEST_RE.search(b):
                continue
            prior = [sp for sp in self.splits if sp[0] < c.lineno and b in sp[1]
                     and not self.reassigned(b, sp[0], c.lineno)]
            if not prior:
                continue   # no split of this data happened before the fit
            split_out = prior[-1][2]
            later_test_eval = any(
                d.lineno > c.lineno and (method(d) in EVAL_METHODS or func_name(d) in METRICS)
                and any(TEST_RE.search(n) for n in names_in(d) & split_out) for d in self.calls)
            if later_test_eval:
                self.add("ML005", c.lineno,
                         why=(f"The model is fitted on `{b}`, which still contains the test rows, and is "
                              "evaluated on the test set afterwards. It has already seen those rows, so the "
                              "test score measures memory, not generalisation."),
                         fix=("Fit on the training split only:\n"
                              "model.fit(X_train, y_train)\n"
                              "Refit on all data only after the final evaluation (for deployment)."))

    def r_outside_cv(self):
        cv_uses = []
        for c in self.calls:
            if func_name(c) in CV_FUNCS:
                data = c.args[1] if len(c.args) > 1 else kw(c, "X")
                if data is not None:
                    cv_uses.append((c, data))
            elif method(c) == "fit" and self.kind(self.receiver_class(c)) == "searcher" and c.args:
                cv_uses.append((c, c.args[0]))
        if not cv_uses:
            return
        for fit, k, cls, arg in self.preprocessing_fits():
            if fit.lineno in getattr(self, "leaky_fit_lines", set()):
                continue
            for cv, data in cv_uses:
                if cv.lineno <= fit.lineno:
                    continue
                db = base_name(data)
                if db not in self.taint_from(self.fit_outputs(fit), fit.lineno, cv.lineno):
                    continue
                full = not ((db and TRAIN_RE.search(db)) or TRAIN_RE.search(base_name(arg) or ""))
                sev = "critical" if (full or k == "resampler") else "medium"
                self.add("ML006", fit.lineno, severity=sev,
                         why=(f"{cls} is fitted on `{base_name(arg)}` before cross-validation on `{db}`. "
                              "Each validation fold was already seen by the preprocessing step, so CV "
                              "scores are optimistic" + (" (and with resampling, synthetic copies of "
                                                         "validation rows sit in the training folds)."
                                                         if k == "resampler" else ".")),
                         fix=("Put the preprocessing inside a Pipeline and cross-validate the pipeline:\n"
                              f"pipe = make_pipeline({cls if cls[0].isupper() else 'StandardScaler'}(), model)"
                              "   # imblearn.pipeline.make_pipeline for samplers\n"
                              "scores = cross_val_score(pipe, X_train, y_train, cv=5)"))
                break

    def r_target_encoding(self):
        if not self.target_cols:
            return
        for c in self.calls:
            if "groupby" not in dotted(c.func) or method(c) not in {"transform", "mean", "agg", "aggregate",
                                                                      "median", "sum"}:
                continue
            if not (strings_in(c) & self.target_cols):
                continue
            b = base_name(c.func)
            if b and TRAIN_RE.search(b) and self.first_split and c.lineno > self.first_split:
                continue
            before_split = self.first_split is None or c.lineno < self.first_split
            if not self.splits and not (self.func_names & CV_FUNCS):
                continue
            if before_split or b in self.split_sources:
                col = sorted(strings_in(c) & self.target_cols)[0]
                self.add("ML007", c.lineno,
                         why=(f"A per-group mean of the target '{col}' is computed on `{b}` before the "
                              "evaluation rows are separated. Each row's encoding includes its own label, "
                              "and validation rows feed their labels into training features."),
                         fix=("Compute the encoding on the training rows only (or out-of-fold), e.g. "
                              "sklearn.preprocessing.TargetEncoder inside a Pipeline:\n"
                              "pipe = make_pipeline(TargetEncoder(), model)\n"
                              "pipe.fit(X_train, y_train)"))

    def r_test_for_selection(self):
        early = bool(re.search(r"early_stopping|EarlyStopping|early_stop", self.code))
        for c in self.calls:
            for k in c.keywords:
                if k.arg not in {"eval_set", "validation_data", "valid_sets", "eval_sets", "X_val"}:
                    continue
                hit = sorted(n for n in names_in(k.value) if STRICT_TEST_RE.search(n))
                if hit:
                    self.add("ML008", c.lineno, severity="high" if early else "medium",
                             why=(f"`{hit[0]}` is passed as `{k.arg}`. "
                                  + ("With early stopping, the number of training rounds is chosen on the "
                                     "test set, so the test score is optimistic."
                                     if early else "Watching the test set while training invites tuning on "
                                     "it; the test score is no longer an unbiased estimate.")),
                             fix=("Carve a validation split out of the training data for monitoring:\n"
                                  "X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)\n"
                                  "model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])\n"
                                  "# touch X_test once, at the very end"))
        for c in self.calls:
            if method(c) == "fit" and self.kind(self.receiver_class(c)) == "searcher" and c.args:
                b = base_name(c.args[0])
                if b and STRICT_TEST_RE.search(b):
                    self.add("ML008", c.lineno,
                             why="Hyper-parameters are tuned on the test set; the reported score is biased upwards.",
                             fix="Run the search on the training data only:\nsearch.fit(X_train, y_train)")

    def r_train_only_metrics(self):
        evals = []
        for c in self.calls:
            fn = func_name(c)
            if fn in METRICS or (method(c) in {"score", "evaluate"} and len(c.args) >= 2):
                ns = names_in(c)
                evals.append((c, any(TEST_RE.search(n) for n in ns), any(TRAIN_RE.search(n) for n in ns)))
        if not evals or any(t for _, t, _ in evals):
            return
        if self.func_names & CV_FUNCS or any(self.kind(t) == "searcher" for t in self.var_types.values()):
            return
        train_evals = [c for c, _, tr in evals if tr]
        if train_evals:
            self.add("ML009", train_evals[0].lineno,
                     why=("Every metric in this code is computed on training data. Training scores show how "
                          "well the model memorised the data, not how it performs on new data."),
                     fix=("Evaluate on held-out data:\n"
                          "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)\n"
                          "model.fit(X_train, y_train)\n"
                          "print(f1_score(y_test, model.predict(X_test)))"))

    def r_combined_fit(self):
        combined = {}
        for c in self.calls:
            if func_name(c) in {"concat", "concatenate", "vstack", "append", "row_stack"}:
                ns = names_in(c)
                if any(TRAIN_RE.search(n) for n in ns) and any(STRICT_TEST_RE.search(n) for n in ns):
                    for t in self.call_targets.get(id(c), set()):
                        combined[t] = c.lineno
        for c in self.calls:
            if method(c) not in {"fit", "fit_transform"} or not c.args:
                continue
            k = self.kind(self.receiver_class(c))
            if k not in {"transformer", "unknown"}:
                continue
            arg = c.args[0]
            b = base_name(arg)
            ns = names_in(arg)
            inline = any(TRAIN_RE.search(n) for n in ns) and any(STRICT_TEST_RE.search(n) for n in ns) \
                and isinstance(arg, ast.Call)
            if (b in combined and combined[b] <= c.lineno) or inline:
                self.add("ML010", c.lineno, confidence="medium",
                         why=("The preprocessing step is fitted on train and test rows together. The test "
                              "distribution (scales, vocabulary, categories) shapes the features the model "
                              "trains on, so offline scores look better than they will on truly unseen data."),
                         fix=("Fit on the training rows, then transform both:\n"
                              "vec.fit(train['text'])\n"
                              "X_train = vec.transform(train['text']); X_test = vec.transform(test['text'])\n"
                              "# categories unseen in training: OneHotEncoder(handle_unknown='ignore')"))

    def r_fillna_stats(self):
        for c in self.calls:
            if method(c) != "fillna":
                continue
            parts = list(c.args) + [k.value for k in c.keywords]
            stats = any(isinstance(n, ast.Call) and method(n) in {"mean", "median", "mode"}
                        for p in parts for n in ast.walk(p))
            if not stats or not self.splits:
                continue
            b = base_name(c.func)
            if not b:
                continue
            leak = False
            if self.first_split and c.lineno < self.first_split:
                tg = self.call_targets.get(id(c), set())
                leak = bool(tg and self.taint_from(tg, c.lineno, max(s[0] for s in self.splits)) & self.split_sources)
            elif b in self.split_sources and not TRAIN_RE.search(b) and not TEST_RE.search(b):
                leak = bool(self.call_targets.get(id(c)))
            if leak:
                self.add("ML011", c.lineno,
                         why=("Missing values are filled with a mean/median/mode computed over all rows, "
                              "including future test rows. It is a small leak, but it shifts the training "
                              "features toward the test distribution."),
                         fix=("Use an imputer fitted on the training rows only (ideally in a Pipeline):\n"
                              "pipe = make_pipeline(SimpleImputer(strategy='median'), model)\n"
                              "pipe.fit(X_train, y_train)"))

    def r_accuracy_only(self):
        if not self.is_classification():
            return
        acc_lines, other = [], False
        for c in self.calls:
            fn, m = func_name(c), method(c)
            if fn == "accuracy_score":
                acc_lines.append(c.lineno)
            elif m == "score" and CLASSIFIER_RE.search(self.receiver_class(c) or "") and len(c.args) >= 2:
                acc_lines.append(c.lineno)
            if fn in CLASS_METRICS_OTHER:
                other = True
            sc = kw(c, "scoring")
            if sc is not None:
                s = strings_in(sc)
                if s and s <= {"accuracy"}:
                    acc_lines.append(c.lineno)
                elif s:
                    other = True
            mt = kw(c, "metrics")
            if mt is not None and any(re.search(r"auc|precision|recall|f1", s, re.I) for s in strings_in(mt)):
                other = True
            if mt is not None and any(re.search(r"AUC|Precision|Recall|F1", dotted(n))
                                      for n in ast.walk(mt) if isinstance(n, ast.Call)):
                other = True
        if acc_lines and not other:
            imbalanced = bool(IMBALANCE_RE.search(self.code))
            self.add("ML012", min(acc_lines), severity="high" if imbalanced else "medium",
                     confidence="high" if imbalanced else "medium",
                     why=("Accuracy is the only metric reported. On imbalanced data a model that always "
                          "predicts the majority class gets high accuracy while catching none of the cases "
                          "that matter" + (" (this code shows signs of class imbalance)." if imbalanced else ".")),
                     fix=("Report metrics that respect class balance:\n"
                          "print(classification_report(y_test, y_pred))\n"
                          "print(balanced_accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))\n"
                          "print(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))"))

    def r_proba_metrics(self):
        for c in self.calls:
            fn = func_name(c)
            if fn not in PROBA_METRICS:
                continue
            parts = list(c.args) + [k.value for k in c.keywords]
            if any(isinstance(p, ast.Call) and method(p) == "predict" for p in parts) or any(
                    isinstance(n, ast.Name) and re.search(r"pred(s|ictions?)?$|y_hat", n.id) and
                    not re.search(r"prob|proba|score", n.id) for p in parts[1:2] for n in [p]):
                self.add("ML013", c.lineno,
                         why=(f"`{fn}` needs probabilities or scores, but it receives hard 0/1 predictions. "
                              "The curve collapses to a single point, so the value is wrong and usually lower "
                              "than the model's real ranking quality."),
                         fix=(f"Pass the positive-class probability:\n"
                              f"{fn}(y_test, model.predict_proba(X_test)[:, 1])"))

    def r_time_split(self):
        time_hint = bool(self.func_names & {"to_datetime", "date_range"}) or \
            any(kw(c, "parse_dates") is not None for c in self.calls) or \
            bool(self.methods & {"shift", "rolling", "resample", "ewm", "diff"} and
                 (self.func_names & {"to_datetime"} or "DatetimeIndex" in self.attr_names
                  or any(re.search(r"date|time", s, re.I) for s in self.strings)))
        if not time_hint or "TimeSeriesSplit" in self.attr_names:
            return
        for c in self.calls:
            fn = func_name(c)
            if fn == "train_test_split" and not is_false(kw(c, "shuffle")):
                line = c.lineno
            elif fn in RANDOM_SPLITTERS and (fn not in {"KFold", "StratifiedKFold"} or
                                             (kw(c, "shuffle") is not None and not is_false(kw(c, "shuffle")))):
                line = c.lineno
            else:
                continue
            self.add("ML014", line, confidence="medium",
                     why=("The data has a time dimension (dates/lag features), but rows are split randomly. "
                          "The model trains on the future and is tested on the past, which overstates how "
                          "well it will forecast."),
                     fix=("Split by time instead:\n"
                          "df = df.sort_values('date')\n"
                          "X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)\n"
                          "# or: cv = TimeSeriesSplit(n_splits=5)"))
            return

    def r_group_split(self):
        if self.attr_names & GROUP_SPLITTERS or any(kw(c, "groups") is not None for c in self.calls):
            return
        ids = set()
        for c in self.calls:
            if method(c) in {"groupby", "merge", "join", "value_counts", "duplicated", "nunique"}:
                ids |= {s for s in strings_in(c) if ID_RE.match(s)}
        if not ids:
            return
        for c in self.calls:
            if func_name(c) in {"train_test_split"} | RANDOM_SPLITTERS:
                idname = sorted(ids)[0]
                self.add("ML015", c.lineno, confidence="low",
                         why=(f"Rows are grouped by '{idname}', so one entity may have several rows, but the "
                              "split is random by row. The same entity can appear in train and test, and the "
                              "model is rewarded for recognising it rather than generalising."),
                         fix=("If an id has more than one row, split by group:\n"
                              "from sklearn.model_selection import GroupShuffleSplit\n"
                              f"gss = GroupShuffleSplit(test_size=0.2, random_state=42)\n"
                              f"train_idx, test_idx = next(gss.split(X, y, groups=df['{idname}']))\n"
                              "# for CV: GroupKFold / StratifiedGroupKFold"))
                return

    def r_torch(self):
        if "torch" not in self.imports or "lightning" in self.imports or \
                "pytorch_lightning" in self.imports or "LightningModule" in self.attr_names:
            return
        calls_by_method = {}
        for c in self.calls:
            calls_by_method.setdefault(method(c), []).append(c)
        if "backward" in calls_by_method and "step" in calls_by_method and "zero_grad" not in self.attr_names:
            line = calls_by_method["backward"][0].lineno
            self.add("ML016", line,
                     why=("`loss.backward()` adds to the existing gradients. Without `optimizer.zero_grad()` "
                          "every step uses the sum of all previous gradients, so training is wrong while "
                          "still looking plausible."),
                     fix=("Reset gradients each step:\n"
                          "optimizer.zero_grad()\n"
                          "loss = criterion(model(xb), yb)\n"
                          "loss.backward()\n"
                          "optimizer.step()"))
        uses_dropout_bn = any(re.search(r"Dropout|BatchNorm", n) for n in self.attr_names)
        evals = [c for c in calls_by_method.get("eval", []) if not c.args]
        train_false = any(c.args and is_false(c.args[0]) for c in calls_by_method.get("train", []))
        if uses_dropout_bn and not evals and not train_false:
            line = min((c.lineno for c in self.calls if re.search(r"Dropout|BatchNorm", dotted(c.func))),
                       default=1)
            self.add("ML017", line,
                     why=("The model uses Dropout/BatchNorm but `model.eval()` is never called. Evaluation "
                          "then runs with dropout active and batch statistics changing, so validation scores "
                          "are noisy and too low, and predictions are not deterministic."),
                     fix=("Switch modes around evaluation:\n"
                          "model.eval()\n"
                          "with torch.no_grad():\n"
                          "    preds = model(X_val)\n"
                          "model.train()"))

    def r_seeds(self):
        items = []
        for c in self.calls:
            fn = func_name(c)
            if has_kwargs_splat(c):
                continue
            seeded = any(kw(c, k) is not None for k in ("random_state", "seed", "random_seed"))
            if fn == "train_test_split" and not seeded and not is_false(kw(c, "shuffle")):
                items.append((c.lineno, "train_test_split(...) has no random_state"))
            elif fn in {"KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold"} and not seeded \
                    and (fn.startswith("Repeated") or (kw(c, "shuffle") is not None and not is_false(kw(c, "shuffle")))):
                items.append((c.lineno, f"{fn}(shuffle=True) has no random_state"))
            elif fn in {"ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit"} and not seeded:
                items.append((c.lineno, f"{fn}(...) has no random_state"))
            elif RANDOM_EST_RE.match(fn) and fn[0].isupper() and not seeded:
                items.append((c.lineno, f"{fn}(...) has no random_state"))
        if "torch" in self.imports and "manual_seed" not in self.attr_names and "seed_everything" not in self.attr_names:
            items.append((self._import_line("torch"), "torch is used but torch.manual_seed(...) is never set"))
        if "tensorflow" in self.imports and not ({"set_seed", "set_random_seed"} & self.attr_names):
            items.append((self._import_line("tensorflow"), "tensorflow is used but tf.random.set_seed(...) is never set"))
        np_random = [c for c in self.calls if re.match(r"(np|numpy)\.random\.(?!seed|default_rng|RandomState|Generator)",
                                                       dotted(c.func))]
        if np_random and not ({"seed"} & self.attr_names) and not any(
                func_name(c) in {"default_rng", "RandomState"} and c.args for c in self.calls):
            items.append((np_random[0].lineno, "np.random is used without np.random.seed(...)"))
        items = [i for i in items if not self.suppressed(i[0], "ML018")]
        if not items:
            return
        items.sort()
        listing = "\n".join(f"- {self.location(ln)}: {msg}" for ln, msg in items[:8])
        more = f"\n- ... and {len(items) - 8} more" if len(items) > 8 else ""
        self.add("ML018", items[0][0],
                 why=("Results change from run to run, so a 'better' score may just be luck and nobody can "
                      "reproduce your numbers. Unseeded randomness found:\n" + listing + more),
                 fix=("Fix the seeds once at the top and pass random_state everywhere:\n"
                      "SEED = 42\n"
                      "np.random.seed(SEED)            # torch.manual_seed(SEED) / tf.random.set_seed(SEED)\n"
                      "train_test_split(X, y, random_state=SEED)\n"
                      "RandomForestClassifier(random_state=SEED)"))

    def _import_line(self, mod) -> int:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Import) and any(a.name.split(".")[0] == mod for a in n.names):
                return n.lineno
            if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] == mod:
                return n.lineno
        return 1

    def r_stratify(self):
        if not self.is_classification():
            return
        if any(MODEL_RE.search(t or "") and not CLASSIFIER_RE.search(t or "") and "Regress" in (t or "")
               for t in self.var_types.values()) and not any(CLASSIFIER_RE.search(t or "") for t in self.var_types.values()):
            return
        for c in self.calls:
            if func_name(c) == "train_test_split" and kw(c, "stratify") is None and not is_false(kw(c, "shuffle")) \
                    and not has_kwargs_splat(c):
                self.add("ML019", c.lineno, confidence="medium",
                         why=("The split is not stratified. With few positive cases, the test set can end up "
                              "with a different class ratio than training (or almost no positives), making "
                              "scores unstable between runs."),
                         fix="train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)")
                return


# ------------------------------------------------------------------- public API
def review_code(code: str, filename: str = "<code>", linemap=None, notes=None) -> Review:
    rv = Review(file=filename, notes=list(notes or []))
    rv.lines_of_code = sum(1 for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    try:
        an = Analyzer(code, linemap)
    except SyntaxError as exc:
        rv.error = f"could not parse the file as Python: {exc.msg} (line {exc.lineno})"
        return rv
    rv.findings = an.run()
    return rv


def review_file(path) -> Review:
    path = Path(path)
    try:
        code, linemap, notes = load_source(path)
    except (OSError, ValueError) as exc:
        return Review(file=str(path), error=f"could not read the file: {exc}")
    return review_code(code, str(path), linemap, notes)


def iter_files(paths):
    skip = {".venv", "venv", "site-packages", "node_modules", ".git", ".ipynb_checkpoints", "__pycache__"}
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in {".py", ".ipynb"} and f.is_file() and not (set(f.parts) & skip):
                    yield f
        else:
            yield p


# --------------------------------------------------------------------- rendering
def _sev_label(s):
    return s.upper()


def split_fix(fix: str):
    """Fix texts start with an explanatory sentence ending in ':' followed by code."""
    first, _, rest = fix.partition("\n")
    if rest and first.rstrip().endswith(":"):
        return first.strip(), rest.strip("\n")
    if "\n" not in fix and not re.search(r"[=(]", fix):
        return fix.strip(), ""
    return "", fix.strip("\n")


def render_markdown(reviews) -> str:
    out = ["# ML Experiment Review", ""]
    for rv in reviews:
        out += [f"## {rv.file}", "", f"**Verdict:** {rv.verdict()}", ""]
        if rv.error:
            out += [f"Not reviewed: {rv.error}", ""]
            continue
        c = rv.counts()
        out += ["| Critical | High | Medium | Low |", "|---|---|---|---|",
                f"| {c['critical']} | {c['high']} | {c['medium']} | {c['low']} |", ""]
        if rv.ai_summary:
            out += [f"**AI reviewer summary:** {rv.ai_summary}", ""]
        for i, f in enumerate(rv.findings, start=1):
            tag = f"{f.rule}" if f.source == "rules" else "AI"
            out += [f"### {i}. [{_sev_label(f.severity)}] {f.title}",
                    f"`{tag}` - {f.location} - confidence: {f.confidence}", ""]
            if f.code:
                out += ["```python", f.code, "```", ""]
            prose, code = split_fix(f.fix)
            out += [f"**Why it matters:** {f.why}", "", f"**Fix:** {prose}".rstrip(), ""]
            if code:
                out += ["```python", code, "```", ""]
            if f.note:
                out += [f"_Note: {f.note}_", ""]
        if rv.dismissed:
            out += ["### Flagged by the rules but doubted by the AI pass (double-check these)", ""]
            for f in rv.dismissed:
                out += [f"- [{_sev_label(f.severity)}] {f.rule} {f.title} ({f.location}): {f.note}"]
            out += [""]
        for n in rv.notes:
            out += [f"> Note: {n}", ""]
        out += ["<details><summary>What static rules cannot see</summary>", "",
                "- Duplicate or near-duplicate rows shared by train and test (check with `pd.merge` on all feature columns).",
                "- Features that are only known after the outcome (e.g. 'days_since_cancellation' when predicting cancellation).",
                "- Leakage that happens in other files, SQL queries or data exports.",
                "- Whether the test set matches the data the model will see in production.",
                "", "</details>", ""]
    return "\n".join(out)


def render_text(reviews) -> str:
    out = []
    for rv in reviews:
        out.append(f"{rv.file}: {rv.verdict()}")
        if rv.error:
            continue
        for f in rv.findings:
            out.append(f"  {f.severity.upper():8} {f.rule if f.source == 'rules' else 'AI':5} "
                       f"{f.location:22} {f.title}")
        if rv.ai_summary:
            out.append(f"  AI summary: {rv.ai_summary}")
        for f in rv.findings:
            if f.note:
                out.append(f"  {f.rule} {f.location}: {f.note}")
        for f in rv.dismissed:
            out.append(f"  doubted by AI: {f.rule} {f.location} - {f.note}")
        for n in rv.notes:
            out.append(f"  note: {n}")
    return "\n".join(out)


def render_json(reviews) -> str:
    return json.dumps({"tool": "mlreview", "version": __version__,
                       "reviews": [{"file": rv.file, "verdict": rv.verdict(), "counts": rv.counts(),
                                    "error": rv.error, "notes": rv.notes, "ai_summary": rv.ai_summary,
                                    "findings": [asdict(f) for f in rv.findings],
                                    "dismissed": [asdict(f) for f in rv.dismissed]} for rv in reviews]},
                      indent=2)


def render(reviews, fmt: str) -> str:
    return {"md": render_markdown, "markdown": render_markdown, "json": render_json,
            "text": render_text}[fmt](reviews)


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def build_parser():
    p = argparse.ArgumentParser(prog="mlreview", description="Find silent bugs in ML training code.")
    p.add_argument("paths", nargs="*", help=".py / .ipynb files or folders")
    p.add_argument("--format", choices=["md", "json", "text"], default="md")
    p.add_argument("--out", help="write the report to this file instead of the screen")
    p.add_argument("--min-severity", choices=SEVERITIES, default="low",
                   help="hide findings below this severity")
    p.add_argument("--fail-on", choices=list(SEVERITIES) + ["none"], default="none",
                   help="exit with code 1 if a finding at or above this severity exists (for CI)")
    p.add_argument("--list-rules", action="store_true", help="print the rule catalogue and exit")
    p.add_argument("--version", action="version", version=f"mlreview {__version__}")
    return p


def filter_and_exit_code(reviews, min_sev, fail_on) -> int:
    for rv in reviews:
        rv.findings = [f for f in rv.findings if SEV_RANK[f.severity] <= SEV_RANK[min_sev]]
    if fail_on == "none":
        return 0
    return 1 if any(SEV_RANK[f.severity] <= SEV_RANK[fail_on] for rv in reviews for f in rv.findings) else 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_rules:
        for rid, (sev, title) in RULES.items():
            safe_print(f"{rid}  {sev:8}  {title}")
        return 0
    if not args.paths:
        build_parser().print_usage()
        print("error: give at least one .py / .ipynb file or folder", file=sys.stderr)
        return 2
    files = list(iter_files(args.paths))
    missing = [str(f) for f in files if not Path(f).exists()]
    if missing:
        print("error: not found: " + ", ".join(missing), file=sys.stderr)
        return 2
    if not files:
        print("error: no .py or .ipynb files found", file=sys.stderr)
        return 2
    reviews = [review_file(f) for f in files]
    code = filter_and_exit_code(reviews, args.min_severity, args.fail_on)
    report = render(reviews, args.format)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        safe_print(render_text(reviews))
        safe_print(f"\nreport written to {args.out}")
    else:
        safe_print(report)
    return code


if __name__ == "__main__":
    sys.exit(main())