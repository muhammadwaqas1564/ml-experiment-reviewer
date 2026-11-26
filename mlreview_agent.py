#!/usr/bin/env python3
"""mlreview_agent - AI second pass on top of the mlreview rules (local use).

1. Runs the deterministic rules (mlreview.py) - fast, repeatable, nothing leaves your machine.
2. Sends the code + rule findings to an LLM on Groq, which
   - doubts rule findings that the code clearly shows are fine (they stay in the report, marked), and
   - adds silent bugs the rules cannot see (e.g. features only known after the outcome).
3. Validates the AI answer (severities, line numbers, schema) and merges it into one report.

Needs GROQ_API_KEY (environment variable or a .env file next to this script or in the
current folder). Without a key, or if the API fails, you still get the rules-only report.

Usage:
    python mlreview_agent.py examples/leaky_churn.py
    python mlreview_agent.py notebook.ipynb --out review.md
    python mlreview_agent.py train.py --model openai/gpt-oss-20b
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlreview as mr  # noqa: E402

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_ADDITIONAL = 10

# Words that show an AI finding is about the same problem as a rule finding on the same line.
RULE_TOPICS = {
    "ML001": ("scal", "normaliz", "imput", "pca", "feature select", "preprocess", "before the split",
              "before split", "fitted on all", "fit on all", "entire dataset"),
    "ML002": ("fit on test", "fitted on test", "fit on the test", "fit on validation", "refit"),
    "ML003": ("target", "label"),
    "ML004": ("smote", "resampl", "oversampl", "undersampl", "synthetic"),
    "ML005": ("full dataset", "all data", "entire data", "whole dataset"),
    "ML006": ("cross-val", "cross val", "cv fold", "fold"),
    "ML007": ("target encod", "mean encod", "groupby"),
    "ML008": ("early stop", "eval_set", "model selection", "tuning", "test set during"),
    "ML009": ("training accuracy", "training score", "train score", "training set"),
    "ML010": ("combined", "concat", "train and test", "train + test", "train+test"),
    "ML011": ("fillna", "missing value", "imput"),
    "ML012": ("accuracy", "metric"),
    "ML013": ("roc", "auc", "log loss", "probabilit"),
    "ML014": ("time", "temporal", "date", "chronolog"),
    "ML015": ("group", "entity", "same user", "same customer", "same patient"),
    "ML016": ("zero_grad", "gradient"),
    "ML017": ("eval()", "eval mode", "dropout", "batchnorm"),
    "ML018": ("seed", "random_state", "reproduc"),
    "ML019": ("stratif",),
}


def restates_rule(title: str, line: int, findings) -> bool:
    t = title.lower()
    return any(f.source == "rules" and f.line == line and any(w in t for w in RULE_TOPICS.get(f.rule, ()))
               for f in findings)
PLACEHOLDER_RE = re.compile(r"^(paste|your|<|changeme|xxx)", re.I)

SYSTEM_PROMPT = """You are a senior machine-learning engineer reviewing a training script for bugs that \
silently ruin results. You receive the code with line numbers and the findings of a static rule engine.

Tasks:
1. For each rule finding, read its explanation and decide if the code clearly shows it is NOT a \
problem. Only then put it in "dismiss" with a one-sentence reason. If unsure, do not dismiss. \
Findings with confidence "high" are almost always right: trace the variables through the whole file \
(for example, a split called "validation" is still the evaluation set if the final score is computed \
on it) before disagreeing.
2. Find additional silent bugs the rules missed, for example: features only known after the outcome \
(target leakage by meaning), duplicate rows shared by train and test, labels misaligned with features \
after shuffling or filtering, evaluation on the wrong split, test data used for any decision, wrong \
loss/activation for the task, wrong metric averaging for multi-class, thresholds tuned on the test set, \
data augmentation applied to validation/test data. Report only issues tied to a specific line of the \
given code. No style comments, no generic advice. Do not add findings that restate a rule finding \
or describe a consequence of it (e.g. "scores are inflated because of the SMOTE issue").

Reply with one JSON object only:
{"summary": "<one or two sentences on how trustworthy the results are>",
 "dismiss": [{"rule": "ML0xx", "line": <int>, "reason": "<one sentence>"}],
 "additional": [{"severity": "critical|high|medium|low", "title": "<short>", "line": <int>,
                 "why": "<why it silently hurts results>", "fix": "<concrete code fix>"}]}"""


class AIError(Exception):
    pass


# ----------------------------------------------------------------------- key loading
def load_api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and not PLACEHOLDER_RE.match(key):
        return key
    for env_file in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(?:export\s+)?GROQ_API_KEY\s*=\s*(.*)$", line)
            if m:
                val = m.group(1).strip().strip("'\"")
                if val and not PLACEHOLDER_RE.match(val):
                    return val
    return None


# ------------------------------------------------------------------------- the call
def urllib_transport(url, headers, body: bytes, timeout):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace"), dict(exc.headers or {})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AIError(f"could not reach the Groq API ({exc})") from exc


def call_llm(messages, model, api_key, transport=urllib_transport, timeout=120):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "User-Agent": f"mlreview/{mr.__version__}"}
    body = {"model": model, "messages": messages, "temperature": 0.1,
            "response_format": {"type": "json_object"}}
    for attempt in range(3):
        status, text, resp_headers = transport(API_URL, headers, json.dumps(body).encode("utf-8"), timeout)
        if status == 200:
            try:
                content = json.loads(text)["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise AIError("unexpected response from the API") from exc
            return parse_json_object(content)
        if status == 400 and "response_format" in body:
            body.pop("response_format")          # some models reject JSON mode: ask in plain text
            continue
        if status == 429 and attempt < 2:
            wait = min(float(resp_headers.get("retry-after", resp_headers.get("Retry-After", 10)) or 10), 30)
            time.sleep(wait)
            continue
        if status == 401:
            raise AIError("the Groq API key was rejected (401). Check GROQ_API_KEY")
        if status == 404:
            raise AIError(f"model '{model}' not found on Groq (404). Try --model openai/gpt-oss-20b")
        raise AIError(f"Groq API returned HTTP {status}")
    raise AIError("Groq API kept refusing the request (rate limit). Try again in a minute")


def parse_json_object(content: str) -> dict:
    content = (content or "").strip()
    try:
        data = json.loads(content)
    except ValueError:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            raise AIError("the AI answer was not JSON")
        try:
            data = json.loads(m.group(0))
        except ValueError as exc:
            raise AIError("the AI answer was not valid JSON") from exc
    if not isinstance(data, dict):
        raise AIError("the AI answer was not a JSON object")
    return data


# ------------------------------------------------------------------- prompt + merge
def build_messages(code: str, review: mr.Review, max_chars: int):
    numbered = "\n".join(f"{i:4d}| {ln}" for i, ln in enumerate(code.splitlines(), start=1))
    truncated = len(numbered) > max_chars
    if truncated:
        numbered = numbered[:max_chars] + "\n... [truncated]"
    findings = [{"rule": f.rule, "line": f.line, "severity": f.severity, "confidence": f.confidence,
                 "title": f.title, "explanation": f.why[:400]} for f in review.findings]
    user = (f"File: {Path(review.file).name}\n\nRule findings:\n{json.dumps(findings, indent=1)}\n\n"
            f"Code:\n```python\n{numbered}\n```")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], truncated


def _clean(text, limit):
    return re.sub(r"\s+\n", "\n", str(text or "")).strip()[:limit]


def merge_ai(review: mr.Review, data: dict, code: str, linemap) -> mr.Review:
    lines = code.splitlines()
    n = len(lines)

    def loc(line):
        if linemap and 0 < line <= len(linemap):
            return f"cell {linemap[line - 1][0]}, line {linemap[line - 1][1]}"
        return f"line {line}"

    review.ai_summary = _clean(data.get("summary"), 600)
    dismiss = data.get("dismiss") or []
    if isinstance(dismiss, list):
        for d in dismiss:
            if not isinstance(d, dict):
                continue
            rule, line = str(d.get("rule", "")).upper(), d.get("line")
            match = next((f for f in review.findings if f.rule == rule and f.line == line), None)
            reason = _clean(d.get("reason"), 300)
            if match and reason:
                if match.confidence == "high":
                    # a high-confidence rule is not overruled by the LLM: keep it, show the disagreement
                    match.note = "AI pass disagrees: " + reason + " (kept: high-confidence rule; check it yourself)"
                else:
                    match.note = "AI pass: " + reason
                    review.findings.remove(match)
                    review.dismissed.append(match)
    extra = data.get("additional") or []
    added = 0
    if isinstance(extra, list):
        for a in extra:
            if added >= MAX_ADDITIONAL or not isinstance(a, dict):
                continue
            sev = str(a.get("severity", "")).lower()
            line = a.get("line")
            title, why, fix = _clean(a.get("title"), 120), _clean(a.get("why"), 900), _clean(a.get("fix"), 900)
            if sev not in mr.SEVERITIES or not isinstance(line, int) or not (1 <= line <= n) or not (title and why):
                continue
            if any(f.line == line and f.title.lower() == title.lower() for f in review.findings):
                continue
            if restates_rule(title, line, review.findings):
                continue   # same line + same topic as a rule finding: a duplicate, not a new bug
            review.findings.append(mr.Finding(
                rule="AI", severity=sev, title=title, line=line, why=why,
                fix=fix or "See explanation.", confidence="medium (AI suggestion - verify)",
                code=lines[line - 1].strip(), location=loc(line), source="ai"))
            added += 1
    review.findings.sort(key=lambda f: (mr.SEV_RANK[f.severity], f.line, f.rule))
    return review


def review_with_ai(path, model, api_key, max_chars=24000, transport=urllib_transport, log=print):
    review = mr.review_file(path)
    if review.error:
        return review
    if not api_key:
        review.notes.append("AI pass skipped: no GROQ_API_KEY found (set it in the environment or a .env file)")
        return review
    code, linemap, _ = mr.load_source(Path(path))
    messages, truncated = build_messages(code, review, max_chars)
    log(f"[ai] sending {Path(path).name} ({len(code.splitlines())} lines) to Groq model {model} ...")
    try:
        data = call_llm(messages, model, api_key, transport=transport)
    except AIError as exc:
        review.notes.append(f"AI pass unavailable: {exc}. Showing the rule findings only.")
        return review
    merge_ai(review, data, code, linemap)
    added = sum(1 for f in review.findings if f.source == "ai")
    disagreed = sum(1 for f in review.findings if f.note.startswith("AI pass disagrees"))
    log(f"[ai] done: {added} new finding(s), {len(review.dismissed)} rule finding(s) doubted, "
        f"{disagreed} disagreement(s) on high-confidence rules")
    if truncated:
        review.notes.append(f"the file was longer than {max_chars} characters; the AI pass saw only the first part")
    return review


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mlreview_agent",
                                description="mlreview rules + an AI second pass (Groq).")
    p.add_argument("paths", nargs="+", help=".py / .ipynb files or folders")
    p.add_argument("--model", default=os.environ.get("GROQ_MODEL", DEFAULT_MODEL))
    p.add_argument("--format", choices=["md", "json", "text"], default="md")
    p.add_argument("--out")
    p.add_argument("--min-severity", choices=mr.SEVERITIES, default="low")
    p.add_argument("--fail-on", choices=list(mr.SEVERITIES) + ["none"], default="none")
    p.add_argument("--max-chars", type=int, default=24000, help="max code characters sent to the AI")
    args = p.parse_args(argv)

    files = list(mr.iter_files(args.paths))
    missing = [str(f) for f in files if not Path(f).exists()]
    if missing or not files:
        print("error: not found: " + (", ".join(missing) or "no .py/.ipynb files"), file=sys.stderr)
        return 2
    key = load_api_key()
    if not key:
        print("[ai] no GROQ_API_KEY found: running the rules only", file=sys.stderr)
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    reviews = [review_with_ai(f, args.model, key, args.max_chars, log=log) for f in files]
    code = mr.filter_and_exit_code(reviews, args.min_severity, args.fail_on)
    report = mr.render(reviews, args.format)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        mr.safe_print(mr.render_text(reviews))
        mr.safe_print(f"\nreport written to {args.out}")
    else:
        mr.safe_print(report)
    return code


if __name__ == "__main__":
    sys.exit(main())