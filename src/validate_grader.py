"""Checks that the answer key and the grader can be trusted.

    python src/validate_grader.py

1. Reference engine
   a. Reproduces real World Cup group tables (2018 Group H, 2022 Groups E and H)
      under the pre-2026 rules, including the 2018 fair-play decision.
   b. Agrees with a second, independently structured implementation of
      FWC26 Article 13 on 5,000 random groups.
   c. Is invariant to the order in which teams and matches are listed.
   d. Reproduces every expected answer in tasks/tasks.jsonl.
2. Grader
   For every task, a battery of synthetic responses: correctly formatted
   variants of the right answer must pass (else false fail); every
   misconception answer, perturbation and malformed answer must fail (else
   false pass).
3. Real outputs (if results/raw exists)
   Grading twice gives byte-identical scores; writes results/grader_audit.csv,
   a seeded random sample of graded responses for manual review.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import (Group, Match, UnresolvedTie, conduct_scores, h2h_key, partition, rank_fwc26,  # noqa: E402
                    rank_legacy_overall_first, rank_thirds, table)
from grade import RAW_DIR, ROOT, grade_text, load_tasks  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


# ------------------------------------------------------------------ 1a real

def real_world_cups() -> int:
    M = Match
    y = lambda team, n: [(1, team, f"#{i}", ["Y"]) for i in range(n)]  # noqa: E731
    cases = [
        ("2018 Group H (Japan above Senegal on fair play)",
         Group("H", ["Colombia", "Japan", "Poland", "Senegal"],
               [M("Colombia", "Japan", 1, 2), M("Poland", "Senegal", 1, 2), M("Japan", "Senegal", 2, 2),
                M("Poland", "Colombia", 0, 3), M("Japan", "Poland", 0, 1), M("Senegal", "Colombia", 0, 1)],
               y("Japan", 4) + y("Senegal", 6) + y("Colombia", 6) + y("Poland", 5), [{}]),
         ["Colombia", "Japan", "Senegal", "Poland"]),
        ("2022 Group E (Spain above Germany on goal difference)",
         Group("E", ["Costa Rica", "Germany", "Japan", "Spain"],
               [M("Germany", "Japan", 1, 2), M("Spain", "Costa Rica", 7, 0), M("Japan", "Costa Rica", 0, 1),
                M("Spain", "Germany", 1, 1), M("Japan", "Spain", 2, 1), M("Costa Rica", "Germany", 2, 4)], [], [{}]),
         ["Japan", "Spain", "Germany", "Costa Rica"]),
        ("2022 Group H (Korea Republic above Uruguay on goals scored)",
         Group("H", ["Ghana", "Korea Republic", "Portugal", "Uruguay"],
               [M("Uruguay", "Korea Republic", 0, 0), M("Portugal", "Ghana", 3, 2), M("Korea Republic", "Ghana", 2, 3),
                M("Portugal", "Uruguay", 2, 0), M("Ghana", "Uruguay", 0, 2), M("Korea Republic", "Portugal", 2, 1)],
               [], [{}]),
         ["Portugal", "Korea Republic", "Uruguay", "Ghana"]),
    ]
    for name, g, expected in cases:
        g.fifa_rankings = []
        got = rank_legacy_overall_first(g)
        check(got == expected, f"real-world check failed: {name}: {got}")
    return len(cases)


# ------------------------------------------------- 1b independent re-implementation

def rank_fwc26_alt(g: Group) -> list[str]:
    """Same rules as engine.rank_fwc26, written as an iterative segment
    refinement instead of recursion, to catch implementation slips."""
    st = table(g.teams, g.matches)
    cs = conduct_scores(g)
    overall = [lambda t: (st[t]["gd"],), lambda t: (st[t]["gf"],), lambda t: (cs[t],)]
    overall += [(lambda t, ed=ed: (-ed[t],)) for ed in g.fifa_rankings]
    segments = [(b, "h2h", 0) for b in partition(g.teams, lambda t: (st[t]["pts"],))]
    while any(len(s[0]) > 1 for s in segments):
        nxt = []
        for teams, mode, idx in segments:
            if len(teams) == 1:
                nxt.append((teams, mode, idx))
            elif mode == "h2h":
                blocks = partition(teams, h2h_key(g, teams))
                nxt += [(teams, "overall", 0)] if len(blocks) == 1 else [(b, "h2h", 0) for b in blocks]
            else:
                if idx >= len(overall):
                    raise UnresolvedTie(teams)
                nxt += [(b, "overall", idx + 1) for b in partition(teams, overall[idx])]
        segments = nxt
    return [s[0][0] for s in segments]


def random_group(rnd: random.Random) -> Group:
    teams = ["T1", "T2", "T3", "T4"]
    pairs = [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]]
    # Low scores make ties (and deep tie-breaks) common.
    matches = [Match(a, b, rnd.choice([0, 0, 1, 1, 2, 3]), rnd.choice([0, 0, 1, 1, 2, 3])) for a, b in pairs]
    discipline = [(1, t, f"#{rnd.randint(1, 23)}", rnd.choice([["Y"], ["Y"], ["Y", "Y"], ["R"], ["Y", "R"]]))
                  for t in teams for _ in range(rnd.randint(0, 2))]
    return Group("X", teams, matches, discipline, [dict(zip(teams, rnd.sample(range(1, 60), 4)))])


def engine_cross_check(n: int = 5000) -> dict:
    rnd = random.Random(2026)
    stats = dict(groups=n, ties=0, reached_step2=0)
    for _ in range(n):
        g = random_group(rnd)
        a, b = rank_fwc26(g), rank_fwc26_alt(g)
        check(a == b, f"engine disagreement on {g.matches}: {a} vs {b}")
        st = table(g.teams, g.matches)
        if len({s["pts"] for s in st.values()}) < 4:
            stats["ties"] += 1
        # permutation invariance
        g2 = Group(g.name, rnd.sample(g.teams, 4), rnd.sample(g.matches, len(g.matches)),
                   rnd.sample(g.discipline, len(g.discipline)), g.fifa_rankings)
        check(rank_fwc26(g2) == a, f"engine depends on input order: {g.matches}")
        if rank_fwc26(g, reapply=False) != a or rank_legacy_overall_first(g) != a:
            stats["reached_step2"] += 1
    return stats


def tasks_reproduce(tasks: dict) -> int:
    for t in tasks.values():
        d = t["data"]
        if t["type"] == "group_ranking":
            g = Group("X", d["teams"], [Match(**m) for m in d["matches"]],
                      [tuple(x) for x in d["discipline"]], d["fifa_rankings"])
            check(rank_fwc26(g) == t["expected"]["ranking"], f"{t['id']}: expected answer not reproduced")
        elif t["type"] == "thirds_ranking":
            check(rank_thirds(d["thirds"])[:8] == t["expected"]["ranking"], f"{t['id']}: expected answer not reproduced")
        else:
            base = [Match(**m) for m in d["matches"]] + [Match(**d["fixed"])]
            other = d["other"]
            good = []
            for letter, (hg, ag) in zip("abcdefgh", d["options"]):
                g = Group("X", d["teams"], base + [Match(other[0], other[1], hg, ag, 3)],
                          [tuple(x) for x in d["discipline"]], d["fifa_rankings"])
                if d["target"] in rank_fwc26(g)[:2]:
                    good.append(letter)
            check(good == t["expected"]["options"], f"{t['id']}: expected answer not reproduced ({good})")
    return len(tasks)


# ------------------------------------------------------------------ 2 grader

def fmt(key: str, value) -> str:
    return f'FINAL ANSWER: {json.dumps({key: value}, ensure_ascii=False)}'


def battery(task: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    key, exp = task["answer_key"], task["expected"][task["answer_key"]]
    reasoning = "Step 1: compute the table. Step 2: apply head-to-head.\n"
    should_pass = [
        ("canonical", reasoning + fmt(key, exp)),
        ("bold marker", reasoning + f"**FINAL ANSWER:** {json.dumps({key: exp})}"),
        ("lowercase names", reasoning + fmt(key, [x.lower() for x in exp])),
        ("upper-case marker + spacing", reasoning + "FINAL  ANSWER :   " + json.dumps({key: exp}, indent=2)),
        ("code fence", reasoning + "```\n" + fmt(key, exp) + "\n```"),
        ("single-quoted literal", reasoning + "Final answer: " + str({key: exp})),
        ("earlier wrong answer revised", reasoning + fmt(key, list(reversed(exp))) + "\nWait, re-checking.\n"
         + fmt(key, exp)),
        ("trailing remark", reasoning + fmt(key, exp) + "\nGood luck to all teams!"),
    ]
    if task["type"] == "scenario":
        should_pass += [("options reordered", reasoning + fmt(key, list(reversed(exp)))),
                        ("options in parentheses", reasoning + fmt(key, [f"({x})" for x in exp]))]
    wrong = []
    for name, alt in task["misconception_answers"].items():
        if alt and alt[key] != exp and (task["type"] != "scenario" or set(alt[key]) != set(exp)):
            wrong.append((f"misconception:{name}", reasoning + fmt(key, alt[key])))
    if task["type"] != "scenario":
        wrong += [("reversed", fmt(key, list(reversed(exp)))),
                  ("adjacent swap", fmt(key, exp[:1] + exp[2:3] + exp[1:2] + exp[3:])),
                  ("missing last", fmt(key, exp[:-1])),
                  ("duplicate", fmt(key, exp[:-1] + exp[:1])),
                  ("extra entry", fmt(key, exp + ["Atlantis"]))]
    else:
        letters = "abcdefgh"[:len(task["data"]["options"])]
        wrong += [("missing one", fmt(key, exp[:-1])),
                  ("one extra", fmt(key, exp + [next(x for x in letters if x not in exp)])),
                  ("empty", fmt(key, []))]
    wrong += [
        ("no marker, prose only", reasoning + "So the order is " + ", ".join(exp) + "."),
        ("correct then revised to wrong", fmt(key, exp) + "\nActually:\n" + fmt(key, exp[:-1])),
        ("wrong key", reasoning + f'FINAL ANSWER: {json.dumps({"answer": exp})}'),
        ("marker without JSON", reasoning + "FINAL ANSWER: " + " > ".join(exp)),
        ("truncated JSON", reasoning + fmt(key, exp)[:-4]),
        ("empty response", ""),
    ]
    return should_pass, wrong


def grader_battery(tasks: dict) -> dict:
    n_pass = n_fail = false_fail = false_pass = 0
    for t in tasks.values():
        good, bad = battery(t)
        for name, text in good:
            n_pass += 1
            if not grade_text(t, text)["correct"]:
                false_fail += 1
                FAILS.append(f"false fail: {t['id']} / {name}")
        for name, text in bad:
            n_fail += 1
            if grade_text(t, text)["correct"]:
                false_pass += 1
                FAILS.append(f"false pass: {t['id']} / {name}")
        # Misconception answers must also be recognised as such.
        for name, alt in t["misconception_answers"].items():
            if alt and alt[t["answer_key"]] != t["expected"][t["answer_key"]]:
                tags = grade_text(t, fmt(t["answer_key"], alt[t["answer_key"]]))["misconceptions"]
                if t["type"] != "scenario" or set(alt["options"]) != set(t["expected"]["options"]):
                    check(name in tags.split("|"), f"{t['id']}: misconception {name} not tagged")
    return dict(should_pass=n_pass, should_fail=n_fail, false_fail=false_fail, false_pass=false_pass)


# ------------------------------------------------------------ 3 real outputs

def real_outputs(tasks: dict) -> dict | None:
    files = sorted(RAW_DIR.rglob("*.json"))
    if not files:
        return None

    def digest() -> str:
        h = hashlib.sha256()
        for p in files:
            rec = json.loads(p.read_text(encoding="utf-8"))
            h.update(json.dumps(grade_text(tasks[rec["task_id"]], rec.get("text") or ""), sort_keys=True).encode())
        return h.hexdigest()

    d1, d2 = digest(), digest()
    check(d1 == d2, "grading is not deterministic")
    rnd = random.Random(7)
    sample = rnd.sample(files, min(40, len(files)))
    out = ROOT / "results" / "grader_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw_file", "task_id", "expected", "grader_parsed", "grader_correct", "final_line_excerpt",
                    "human_correct (fill in 0/1)"])
        for p in sample:
            rec = json.loads(p.read_text(encoding="utf-8"))
            t = tasks[rec["task_id"]]
            res = grade_text(t, rec.get("text") or "")
            text = rec.get("text") or ""
            idx = text.lower().rfind("final answer")
            excerpt = text[idx: idx + 300] if idx >= 0 else text[-300:]
            w.writerow([p.relative_to(ROOT).as_posix(), t["id"], json.dumps(t["expected"][t["answer_key"]]),
                        res["answer"], res["correct"], excerpt.replace("\n", " "), ""])
    return dict(responses=len(files), deterministic=d1 == d2, audit_file=out.relative_to(ROOT).as_posix(),
                audit_rows=len(sample))


def main() -> None:
    tasks = load_tasks()
    print("1a real World Cup groups reproduced:", real_world_cups())
    print("1b/1c engine cross-check:", engine_cross_check())
    print("1d task answers reproduced:", tasks_reproduce(tasks))
    print("2  grader battery:", grader_battery(tasks))
    real = real_outputs(tasks)
    print("3  real outputs:", real if real else "none yet (run src/run_models.py first)")
    if FAILS:
        print(f"\nFAILED ({len(FAILS)}):")
        for f in FAILS[:50]:
            print("  -", f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
