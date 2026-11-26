"""Tests for mlreview. Standard library only:  python -m unittest discover -s tests -v"""
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mlreview as mr  # noqa: E402

EX = ROOT / "examples"


def rules_of(code):
    return {f.rule for f in mr.review_code(textwrap.dedent(code)).findings}


class TestDemoFiles(unittest.TestCase):
    def test_clean_example_has_no_findings(self):
        rv = mr.review_file(EX / "clean_churn.py")
        self.assertEqual(rv.findings, [], [f.rule for f in rv.findings])
        self.assertTrue(rv.verdict().startswith("PASS"))

    def test_leaky_example(self):
        rv = mr.review_file(EX / "leaky_churn.py")
        self.assertEqual({f.rule for f in rv.findings},
                         {"ML001", "ML003", "ML011", "ML013", "ML018", "ML019"})
        self.assertTrue(rv.verdict().startswith("FAIL"))
        self.assertEqual(rv.findings[0].severity, "critical")   # ranked by severity

    def test_kaggle_notebook(self):
        rv = mr.review_file(EX / "leaky_kaggle.ipynb")
        got = {f.rule for f in rv.findings}
        self.assertEqual(got, {"ML003", "ML004", "ML007", "ML008", "ML010", "ML012", "ML015", "ML018", "ML019"})
        smote = next(f for f in rv.findings if f.rule == "ML004")
        self.assertEqual(smote.location, "cell 8, line 1")
        self.assertIn("SMOTE().fit_resample", smote.code)
        self.assertIn("SMOTE(...) has no random_state", next(f for f in rv.findings if f.rule == "ML018").why)

    def test_torch_example(self):
        rv = mr.review_file(EX / "torch_training.py")
        self.assertEqual({f.rule for f in rv.findings}, {"ML016", "ML017", "ML018"})


class TestRulesPositive(unittest.TestCase):
    def test_ml001_scaler_before_split(self):
        self.assertIn("ML001", rules_of("""
            X = scaler_input
            sc = StandardScaler()
            X = sc.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        """))

    def test_ml001_fit_then_transform_variable(self):
        self.assertIn("ML001", rules_of("""
            pca = PCA(n_components=5)
            pca.fit(X)
            X_p = pca.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_p, y, random_state=0)
        """))

    def test_ml001_fit_on_full_after_split(self):
        self.assertIn("ML001", rules_of("""
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            sc = MinMaxScaler().fit(X)
            X_train = sc.transform(X_train)
        """))

    def test_ml001_manual_slice_split(self):
        self.assertIn("ML001", rules_of("""
            X = SelectKBest(k=10).fit_transform(X, y)
            X_train = X[:800]
            X_test = X[800:]
        """))

    def test_ml002_fit_transform_on_test(self):
        self.assertIn("ML002", rules_of("""
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.fit_transform(X_test)
        """))

    def test_ml002_model_fit_on_validation(self):
        self.assertIn("ML002", rules_of("model = LogisticRegression()\nmodel.fit(X_val, y_val)\n"))

    def test_ml003_whole_dataframe(self):
        self.assertIn("ML003", rules_of("y = df['label']\nX = df\n"))

    def test_ml003_drop_other_column(self):
        self.assertIn("ML003", rules_of("y = df.price\nX = df.drop(columns=['id'])\n"))

    def test_ml004_smote_before_split(self):
        self.assertIn("ML004", rules_of("""
            X2, y2 = SMOTE(random_state=0).fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X2, y2, random_state=0)
        """))

    def test_ml005_full_fit_then_test(self):
        self.assertIn("ML005", rules_of("""
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            model = RandomForestRegressor(random_state=0)
            model.fit(X, y)
            print(r2_score(y_test, model.predict(X_test)))
        """))

    def test_ml006_scaler_before_cv(self):
        found = mr.review_code(textwrap.dedent("""
            Xs = StandardScaler().fit_transform(X)
            scores = cross_val_score(LogisticRegression(), Xs, y, cv=5, scoring='f1')
        """)).findings
        f = next(f for f in found if f.rule == "ML006")
        self.assertEqual(f.severity, "critical")

    def test_ml006_train_only_is_medium(self):
        found = mr.review_code(textwrap.dedent("""
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
            X_tr = StandardScaler().fit_transform(X_train)
            print(cross_val_score(LogisticRegression(), X_tr, y_train, cv=5, scoring='f1'))
        """)).findings
        self.assertEqual(next(f for f in found if f.rule == "ML006").severity, "medium")

    def test_ml008_eval_set_test(self):
        self.assertIn("ML008", rules_of("""
            m = LGBMClassifier(random_state=0)
            m.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[early_stopping(50)])
        """))

    def test_ml008_keras_validation_data(self):
        self.assertIn("ML008", rules_of("model.fit(X_train, y_train, validation_data=(X_test, y_test))\n"))

    def test_ml009_train_metrics_only(self):
        self.assertIn("ML009", rules_of("""
            model = Ridge()
            model.fit(X_train, y_train)
            print(mean_squared_error(y_train, model.predict(X_train)))
        """))

    def test_ml011_fillna_mean(self):
        self.assertIn("ML011", rules_of("""
            df['age'] = df['age'].fillna(df['age'].mean())
            X_train, X_test = train_test_split(df, random_state=0)
        """))

    def test_ml012_accuracy_only(self):
        self.assertIn("ML012", rules_of("""
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            print(accuracy_score(y_test, clf.predict(X_test)))
        """))

    def test_ml013_auc_with_predict(self):
        self.assertIn("ML013", rules_of("roc_auc_score(y_test, clf.predict(X_test))\n"))

    def test_ml013_auc_with_pred_variable(self):
        self.assertIn("ML013", rules_of("y_pred = clf.predict(X_test)\nroc_auc_score(y_test, y_pred)\n"))

    def test_ml014_time_series_random_split(self):
        self.assertIn("ML014", rules_of("""
            df['date'] = pd.to_datetime(df['date'])
            df['lag1'] = df['sales'].shift(1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        """))

    def test_ml016_ml017_torch(self):
        got = rules_of("""
            import torch
            torch.manual_seed(0)
            net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8))
            for xb, yb in loader:
                loss = crit(net(xb), yb)
                loss.backward()
                opt.step()
        """)
        self.assertTrue({"ML016", "ML017"} <= got)
        self.assertNotIn("ML018", got)


class TestNoFalsePositives(unittest.TestCase):
    def test_pipeline_with_cv(self):
        self.assertEqual(rules_of("""
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')
        """), set())

    def test_fit_on_train_transform_test(self):
        self.assertEqual(rules_of("""
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            reg = LinearRegression().fit(X_train, y_train)
            print(r2_score(y_test, reg.predict(X_test)))
        """), set())

    def test_target_popped_is_fine(self):
        self.assertNotIn("ML003", rules_of("y = df['label']\ndf = df.drop(columns=['label'])\nX = df\n"))

    def test_target_dropped_with_axis(self):
        self.assertNotIn("ML003", rules_of("y = df['label']\nX = df.drop('label', axis=1)\n"))

    def test_validation_set_for_early_stopping_is_fine(self):
        self.assertNotIn("ML008", rules_of(
            "m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20)\n"))

    def test_final_refit_after_evaluation_is_fine(self):
        self.assertNotIn("ML005", rules_of("""
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            model = Ridge()
            model.fit(X_train, y_train)
            print(r2_score(y_test, model.predict(X_test)))
            model.fit(X, y)
        """))

    def test_auc_with_proba(self):
        self.assertNotIn("ML013", rules_of("roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])\n"))

    def test_torch_correct_loop(self):
        self.assertEqual(rules_of("""
            import torch
            torch.manual_seed(0)
            net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2))
            for xb, yb in loader:
                opt.zero_grad()
                crit(net(xb), yb).backward()
                opt.step()
            net.eval()
        """), set())

    def test_fit_only_for_plotting_or_printing(self):
        self.assertNotIn("ML001", rules_of("""
            pca = PCA(n_components=2).fit(X)
            plt.plot(pca.components_)
            print(SimpleImputer().fit_transform(X))
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        """))

    def test_fitted_on_other_section_data(self):
        self.assertNotIn("ML001", rules_of("""
            sc = StandardScaler()
            sc.fit(X_train_old)
            X, y = load_new()
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            print(sc.transform(X_test))
        """))

    def test_reassigned_object_not_blamed(self):
        self.assertNotIn("ML001", rules_of("""
            pca = PCA(n_components=2).fit(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            pca = pipe.named_steps['pca']
            print(pca.transform(X_test))
        """))

    def test_regression_needs_no_stratify(self):
        self.assertNotIn("ML019", rules_of("""
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            m = RandomForestRegressor(random_state=0).fit(X_train, y_train)
            print(r2_score(y_test, m.predict(X_test)))
        """))


class TestBehaviour(unittest.TestCase):
    def test_suppression_comments(self):
        base = "y = df['t']\nX = df{c}\n"
        self.assertIn("ML003", rules_of(base.format(c="")))
        self.assertNotIn("ML003", rules_of(base.format(c="  # mlreview: ignore")))
        self.assertNotIn("ML003", rules_of(base.format(c="  # noqa: ML003")))
        self.assertIn("ML003", rules_of(base.format(c="  # noqa: ML012")))

    def test_syntax_error_in_py(self):
        rv = mr.review_code("def broken(:\n")
        self.assertIn("could not parse", rv.error)
        self.assertEqual(rv.verdict(), "NOT REVIEWED")

    def test_notebook_bad_cell_is_skipped_not_fatal(self):
        nb = {"cells": [
            {"cell_type": "code", "source": ["y = df['t']\n", "X = df\n"]},
            {"cell_type": "code", "source": "this is not python ((("},
            {"cell_type": "code", "source": "%%time\nmodel.fit(X_test, y_test)"}],
            "nbformat": 4, "nbformat_minor": 5, "metadata": {}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nb.ipynb"
            p.write_text(json.dumps(nb), encoding="utf-8")
            rv = mr.review_file(p)
        self.assertEqual(rv.error, "")
        self.assertTrue(any("cell 2 skipped" in n for n in rv.notes))
        locs = {f.rule: f.location for f in rv.findings}
        self.assertEqual(locs["ML003"], "cell 1, line 2")
        self.assertEqual(locs["ML002"], "cell 3, line 2")

    def test_markdown_report_contents(self):
        md = mr.render_markdown([mr.review_file(EX / "leaky_churn.py")])
        for s in ("**Verdict:** FAIL", "### 1. [CRITICAL]", "**Why it matters:**", "**Fix:**",
                  "What static rules cannot see"):
            self.assertIn(s, md)

    def test_every_finding_has_why_and_fix(self):
        for f in (EX / "leaky_churn.py", EX / "leaky_kaggle.ipynb", EX / "torch_training.py"):
            for fd in mr.review_file(f).findings:
                self.assertTrue(fd.why and fd.fix and fd.rule in mr.RULES and fd.severity in mr.SEVERITIES)


class TestCLI(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "mlreview.py"), *args],
                              capture_output=True, text=True, cwd=ROOT)

    def test_json_output_and_exit_codes(self):
        r = self.run_cli("examples/leaky_churn.py", "--format", "json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data["reviews"][0]["counts"]["critical"], 2)
        self.assertEqual(self.run_cli("examples/leaky_churn.py", "--fail-on", "critical").returncode, 1)
        self.assertEqual(self.run_cli("examples/clean_churn.py", "--fail-on", "low").returncode, 0)

    def test_folder_min_severity_and_out_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.md"
            r = self.run_cli("examples", "--min-severity", "high", "--out", str(out))
            self.assertEqual(r.returncode, 0, r.stderr)
            text = out.read_text(encoding="utf-8")
        self.assertIn("leaky_kaggle.ipynb", text)
        self.assertNotIn("[LOW]", text)
        self.assertNotIn("[MEDIUM]", text)

    def test_missing_file_and_list_rules(self):
        self.assertEqual(self.run_cli("nope.py").returncode, 2)
        r = self.run_cli("--list-rules")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(len(r.stdout.strip().splitlines()), len(mr.RULES))


if __name__ == "__main__":
    unittest.main()
