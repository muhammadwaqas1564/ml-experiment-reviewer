"""Tests for the AI second pass, with a fake Groq API (no key, no network)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mlreview as mr  # noqa: E402
import mlreview_agent as ag  # noqa: E402

LEAKY = ROOT / "examples" / "leaky_churn.py"
NOTEBOOK = ROOT / "examples" / "leaky_kaggle.ipynb"


def fake_api(*responses):
    """Transport returning queued (status, content_or_body) pairs and recording requests."""
    queue, seen = list(responses), []

    def transport(url, headers, body, timeout):
        seen.append(json.loads(body))
        status, payload = queue.pop(0)
        if status == 200 and isinstance(payload, (dict, str)):
            content = payload if isinstance(payload, str) else json.dumps(payload)
            return 200, json.dumps({"choices": [{"message": {"content": content}}]}), {}
        return status, json.dumps({"error": {"message": "x"}}), {"retry-after": "0"}
    transport.seen = seen
    return transport


GOOD = {
    "summary": "Scores are inflated by leakage.",
    "dismiss": [{"rule": "ML019", "line": 29, "reason": "The data is balanced enough in this demo."},
                {"rule": "ML999", "line": 1, "reason": "unknown rule must be ignored"}],
    "additional": [
        {"severity": "medium", "title": "Class imbalance ignored by the model", "line": 31,
         "why": "Only 5% churners and no class_weight.", "fix": "RandomForestClassifier(class_weight='balanced')"},
        {"severity": "urgent", "title": "bad severity", "line": 3, "why": "x", "fix": "x"},
        {"severity": "low", "title": "line out of range", "line": 9999, "why": "x", "fix": "x"},
        {"severity": "low", "title": "line not an int", "line": "7", "why": "x", "fix": "x"},
    ],
}


class TestAgent(unittest.TestCase):
    def test_merge_dismiss_and_validate(self):
        t = fake_api((200, GOOD))
        rv = ag.review_with_ai(LEAKY, "m", "k", transport=t, log=lambda m: None)
        rules = [f.rule for f in rv.findings]
        self.assertNotIn("ML019", rules)                        # dismissed ...
        self.assertEqual([f.rule for f in rv.dismissed], ["ML019"])   # ... but still reported
        ai = [f for f in rv.findings if f.source == "ai"]
        self.assertEqual(len(ai), 1)                            # invalid AI items dropped
        self.assertEqual(ai[0].location, "line 31")
        self.assertIn("RandomForestClassifier", ai[0].code)
        self.assertEqual(rv.ai_summary, "Scores are inflated by leakage.")
        md = mr.render_markdown([rv])
        self.assertIn("AI reviewer summary", md)
        self.assertIn("doubted by the AI pass", md)
        req = t.seen[0]
        self.assertEqual(req["response_format"], {"type": "json_object"})
        self.assertIn("  24| X = df.drop", req["messages"][1]["content"])

    def test_high_confidence_rule_is_not_overruled(self):
        code, _, _ = mr.load_source(NOTEBOOK)
        line = next(i for i, l in enumerate(code.splitlines(), 1) if "eval_set" in l)
        t = fake_api((200, {"summary": "s", "dismiss": [
            {"rule": "ML008", "line": line, "reason": "X_test is only a validation split."}]}))
        rv = ag.review_with_ai(NOTEBOOK, "m", "k", transport=t, log=lambda m: None)
        kept = next(f for f in rv.findings if f.rule == "ML008")
        self.assertTrue(kept.note.startswith("AI pass disagrees"))
        self.assertEqual(rv.dismissed, [])
        self.assertIn("AI pass disagrees", mr.render_text([rv]))
        self.assertIn('"explanation"', t.seen[0]["messages"][1]["content"])

    def test_duplicates_of_rule_findings_are_dropped(self):
        code, _, _ = mr.load_source(NOTEBOOK)
        lines = code.splitlines()
        x_line = next(i for i, l in enumerate(lines, 1) if 'select_dtypes' in l)
        smote_line = next(i for i, l in enumerate(lines, 1) if 'fit_resample' in l)
        t = fake_api((200, {"summary": "s", "additional": [
            {"severity": "critical", "title": "Target column still present in features", "line": x_line,
             "why": "w", "fix": "f"},
            {"severity": "critical", "title": "SMOTE applied on leaked data", "line": smote_line,
             "why": "w", "fix": "f"},
            {"severity": "high", "title": "Entity ID leakage via user_id numeric feature", "line": x_line,
             "why": "user_id is numeric and kept as a feature", "fix": "drop user_id"}]}))
        rv = ag.review_with_ai(NOTEBOOK, "m", "k", transport=t, log=lambda m: None)
        ai = [f.title for f in rv.findings if f.source == "ai"]
        self.assertEqual(ai, ["Entity ID leakage via user_id numeric feature"])

    def test_json_mode_rejected_falls_back(self):
        t = fake_api((400, None), (200, "Here you go:\n" + json.dumps({"summary": "ok"}) + "\nthanks"))
        rv = ag.review_with_ai(LEAKY, "m", "k", transport=t, log=lambda m: None)
        self.assertEqual(rv.ai_summary, "ok")
        self.assertNotIn("response_format", t.seen[1])

    def test_rate_limit_retry(self):
        t = fake_api((429, None), (200, {"summary": "after retry"}))
        with mock.patch.object(ag.time, "sleep") as sleep:
            rv = ag.review_with_ai(LEAKY, "m", "k", transport=t, log=lambda m: None)
        self.assertEqual(rv.ai_summary, "after retry")
        sleep.assert_called_once()

    def test_bad_key_keeps_rule_report(self):
        rv = ag.review_with_ai(LEAKY, "m", "k", transport=fake_api((401, None)), log=lambda m: None)
        self.assertTrue(any("rejected" in n for n in rv.notes))
        self.assertEqual(len(rv.findings), 6)

    def test_network_error_keeps_rule_report(self):
        def broken(*a, **k):
            raise ag.AIError("could not reach the Groq API (offline)")
        rv = ag.review_with_ai(LEAKY, "m", "k", transport=broken, log=lambda m: None)
        self.assertTrue(any("unavailable" in n for n in rv.notes))
        self.assertEqual(len(rv.findings), 6)

    def test_no_key(self):
        rv = ag.review_with_ai(LEAKY, "m", None, log=lambda m: None)
        self.assertTrue(any("no GROQ_API_KEY" in n for n in rv.notes))

    def test_notebook_locations_for_ai_findings(self):
        code, linemap, _ = mr.load_source(NOTEBOOK)
        line = next(i for i, l in enumerate(code.splitlines(), 1) if "eval_set" in l)
        t = fake_api((200, {"summary": "s", "additional": [
            {"severity": "medium", "title": "x", "line": line, "why": "w", "fix": "f"}]}))
        rv = ag.review_with_ai(NOTEBOOK, "m", "k", transport=t, log=lambda m: None)
        self.assertEqual(next(f for f in rv.findings if f.source == "ai").location, "cell 9, line 2")

    def test_garbage_answer(self):
        rv = ag.review_with_ai(LEAKY, "m", "k", transport=fake_api((200, "no json here")), log=lambda m: None)
        self.assertTrue(any("not JSON" in n for n in rv.notes))

    def test_key_loading_ignores_placeholders(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=True):
            old = os.getcwd()
            os.chdir(d)
            try:
                Path(".env").write_text("GROQ_API_KEY=PASTE_YOUR_KEY_HERE\n", encoding="utf-8")
                with mock.patch.object(ag, "__file__", str(Path(d) / "x.py")):
                    self.assertIsNone(ag.load_api_key())
                    Path(".env").write_text('GROQ_API_KEY="abc123"\n', encoding="utf-8")
                    self.assertEqual(ag.load_api_key(), "abc123")
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()