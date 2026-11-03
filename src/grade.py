"""Deterministic grader. No LLM judge is involved.

Each response must end with `FINAL ANSWER: {"<key>": [...]}`. The grader
reads the last such line, normalises team names / option letters, and compares
with the expected answer:

  group_ranking   exact 1st-4th order            partial = share of positions right
  thirds_ranking  exact order of the 8 qualifiers partial = share of the 8 qualifiers right (as a set)
  scenario        exact set of option letters     partial = Jaccard similarity

`correct` (0/1) is the headline metric; `partial` is reported for context only.
A wrong answer that equals the output of a known misconception (see
engine.GROUP_VARIANTS / THIRD_VARIANTS) is tagged with that misconception.

    python src/grade.py            # grades results/raw/** -> results/scores.csv
"""
from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "tasks" / "tasks.jsonl"
RAW_DIR = ROOT / "results" / "raw"
SCORES_PATH = ROOT / "results" / "scores.csv"

MARKER = re.compile(r"final\s+answer", re.IGNORECASE)


def load_tasks() -> dict[str, dict]:
    with TASKS_PATH.open(encoding="utf-8") as f:
        return {t["id"]: t for t in map(json.loads, f)}


def _decode_object(text: str, start: int):
    """Decode the object that starts at text[start] ('{'). Accepts strict JSON
    and Python-literal style (single quotes); returns None if neither parses."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        pass
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return ast.literal_eval(text[start:i + 1])
                except (ValueError, SyntaxError):
                    return None
    return None


def extract_answer(text: str, key: str) -> list | None:
    """Value of `key` in the object after the LAST 'FINAL ANSWER' marker.
    Answers without the marker are not accepted (counted as format failures)."""
    if not text:
        return None
    markers = list(MARKER.finditer(text))
    if not markers:
        return None
    tail_start = markers[-1].end()
    brace = text.find("{", tail_start)
    if brace == -1:
        return None
    obj = _decode_object(text, brace)
    if not isinstance(obj, dict):
        return None
    value = next((v for k, v in obj.items() if str(k).strip().lower() == key), None)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return None
    return value


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip(".*`'\"").strip().casefold()


def normalise(task: dict, answer: list[str]) -> list[str]:
    if task["type"] == "scenario":
        return [_norm(x).strip("()") for x in answer]
    canon: dict[str, str] = {}
    names = task["data"].get("teams") or [r["team"] for r in task["data"]["thirds"]]
    if task["type"] == "thirds_ranking":
        names = names + task["data"].get("group_l_teams", [])
    for n in names:
        canon[_norm(n)] = n
    return [canon.get(_norm(x), x.strip()) for x in answer]


def score(task: dict, answer: list[str] | None) -> dict:
    key = task["answer_key"]
    expected = task["expected"][key]
    if answer is None:
        return dict(parse_ok=0, correct=0, partial=0.0, answer="", misconceptions="")
    got = normalise(task, answer)
    if task["type"] == "scenario":
        g, e = set(got), set(expected)
        correct = g == e
        partial = len(g & e) / len(g | e) if g | e else 1.0
    elif task["type"] == "thirds_ranking":
        correct = got == expected
        partial = len(set(got) & set(expected)) / len(expected)
    else:
        correct = got == expected
        partial = sum(a == b for a, b in zip(got, expected)) / len(expected)
    matched = []
    if not correct:
        for name, alt in task.get("misconception_answers", {}).items():
            if not alt:
                continue
            alt_v = alt[key]
            same = set(got) == set(alt_v) if task["type"] == "scenario" else got == alt_v
            if same and alt_v != expected:
                matched.append(name)
    return dict(parse_ok=1, correct=int(correct), partial=round(partial, 4),
                answer=json.dumps(got, ensure_ascii=False), misconceptions="|".join(matched))


def grade_text(task: dict, text: str) -> dict:
    return score(task, extract_answer(text, task["answer_key"]))


FIELDS = ["model", "requested_model", "response_model", "condition", "task_id", "category", "difficulty",
          "sample", "parse_ok", "correct", "partial", "misconceptions", "answer", "error", "stop_reason",
          "output_tokens", "raw_file"]


def main() -> None:
    tasks = load_tasks()
    rows = []
    for path in sorted(RAW_DIR.rglob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        task = tasks[rec["task_id"]]
        res = grade_text(task, rec.get("text") or "")
        rows.append(dict(
            model=rec["model_label"], requested_model=rec["requested_model"],
            response_model=rec.get("response_model", ""), condition=rec["condition"],
            task_id=rec["task_id"], category=task["category"], difficulty=task["difficulty"],
            sample=rec["sample"], **res, error=rec.get("error") or "", stop_reason=rec.get("stop_reason") or "",
            output_tokens=(rec.get("usage") or {}).get("output_tokens", ""),
            raw_file=path.relative_to(ROOT).as_posix()))
    rows.sort(key=lambda r: (r["model"], r["condition"], r["task_id"], int(r["sample"])))
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCORES_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else 0
    print(f"graded {n} responses -> {SCORES_PATH.relative_to(ROOT)} (overall accuracy {acc:.3f})")


if __name__ == "__main__":
    main()
