"""Builds tasks/tasks.jsonl from the scenario definitions below.

Every scenario carries a hand-written expected answer and derivation. The
build fails unless the reference engine (src/engine.py) produces exactly the
same answer, so each expected answer is checked twice: once by hand against the
regulation text and once by code.

    python src/build_tasks.py
"""
from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

from engine import (GROUP_VARIANTS, THIRD_VARIANTS, Group, Match, UnresolvedTie,
                    conduct_scores, rank_fwc26, rank_legacy_overall_first, rank_thirds, table)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tasks" / "tasks.jsonl"
ARTICLE_13 = (ROOT / "prompts" / "fwc26_article13.txt").read_text(encoding="utf-8").strip()

FORMAT_HEADER = (
    "Competition: FIFA World Cup 26 (48 teams in 12 groups of four; the group winners, "
    "the runners-up and the eight best third-placed teams advance to the round of 32).\n"
    "The teams and results below are hypothetical."
)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def M(md: int, home: str, hg: int, ag: int, away: str) -> Match:
    return Match(home, away, hg, ag, md)


def seed_for(task_id: str) -> random.Random:
    return random.Random(zlib.crc32(task_id.encode()))


def background_discipline(task_id: str, teams: list[str], matchdays: int = 3) -> list:
    """A few single yellow cards per team so every prompt has a disciplinary
    record (its presence must not hint that conduct decides the task)."""
    rnd = seed_for(task_id + "/cards")
    out = []
    for t in teams:
        players = rnd.sample(range(2, 24), 3)
        for i in range(rnd.randint(1, 3)):
            out.append((rnd.randint(1, matchdays), t, f"#{players[i]}", ["Y"]))
    return out


def background_rankings(task_id: str, teams: list[str]) -> list[dict[str, int]]:
    rnd = seed_for(task_id + "/rank")
    return [dict(zip(teams, rnd.sample(range(6, 90), len(teams))))]


# ----------------------------------------------------------------- rendering

def describe_cards(cards: list[str]) -> str:
    y, r = cards.count("Y"), cards.count("R")
    if y >= 2 and not r:
        return "yellow card, then a second yellow card (sent off)"
    if y and r:
        return "yellow card, then a direct red card (sent off)"
    if r:
        return "direct red card (sent off)"
    return "yellow card"


def render_group_block(g: Group, *, with_cards: bool = True, with_ranking: bool = True) -> str:
    lines = [f"Group {g.name}: {', '.join(g.teams)}", "", "Results:"]
    for md in sorted({m.matchday for m in g.matches}):
        games = "; ".join(f"{m.home} {m.hg}-{m.ag} {m.away}" for m in g.matches if m.matchday == md)
        lines.append(f"Matchday {md}: {games}")
    if with_cards:
        lines += ["", "Disciplinary record (every card shown to a player or team official in the group stage):"]
        for t in g.teams:
            entries = sorted((d for d in g.discipline if d[1] == t), key=lambda d: (d[0], d[2]))
            if not entries:
                lines.append(f"- {t}: no cards")
                continue
            desc = "; ".join(f"matchday {md}, {who if not who.startswith('#') else 'player ' + who}: {describe_cards(c)}"
                             for md, _t, who, c in entries)
            lines.append(f"- {t}: {desc}")
    if with_ranking:
        labels = ["most recent published edition", "edition immediately preceding the most recent one"]
        for i, ed in enumerate(g.fifa_rankings):
            lines += ["", f"FIFA/Coca-Cola Men's World Ranking, {labels[i]} (position):"]
            pos = sorted(ed.items(), key=lambda kv: kv[1])
            seen: dict[int, int] = {}
            for _t, p in pos:
                seen[p] = seen.get(p, 0) + 1
            lines.append(", ".join(f"{t} {'joint ' if seen[p] > 1 else ''}{ordinal(p)}" for t, p in pos))
    return "\n".join(lines)


def answer_instruction(key: str, example: str) -> str:
    return ("Explain your reasoning, then finish with exactly one final line in this format:\n"
            f'FINAL ANSWER: {{"{key}": {example}}}')


def make_prompts(body: str, question: str, key: str, example: str) -> dict[str, str]:
    closed = f"{FORMAT_HEADER}\n\n{body}\n\nQuestion: {question}\n\n{answer_instruction(key, example)}"
    open_ = (f"{FORMAT_HEADER}\n\n{body}\n\nQuestion: {question}\n\n"
             f"Use the regulation text below.\n\n<regulation>\n{ARTICLE_13}\n</regulation>\n\n"
             f"{answer_instruction(key, example)}")
    return {"closed_book": closed, "open_book": open_}


def variants_for_group(g: Group) -> dict[str, list[str] | None]:
    out = {}
    for name, fn in GROUP_VARIANTS.items():
        try:
            out[name] = fn(g)
        except UnresolvedTie:
            out[name] = None
    return out


TASKS: list[dict] = []


def group_task(tid: str, category: str, difficulty: str, name: str, teams: list[str],
               matches: list[Match], expected: list[str], derivation: str, tests: str,
               discipline: list | None = None, rankings: list | None = None) -> Group:
    g = Group(name, teams, matches,
              discipline if discipline is not None else background_discipline(tid, teams),
              rankings if rankings is not None else background_rankings(tid, teams))
    got = rank_fwc26(g)
    assert got == expected, f"{tid}: engine {got} != hand-derived {expected}"
    body = render_group_block(g)
    q = (f"Apply the FIFA World Cup 26 tie-breaking regulations to determine the final ranking "
         f"of Group {name}, from 1st to 4th.")
    example = json.dumps([f"<{ordinal(i)} place>" for i in range(1, 5)])
    st = table(teams, matches)
    TASKS.append(dict(
        id=tid, type="group_ranking", category=category, difficulty=difficulty,
        what_it_tests=tests, answer_key="ranking",
        prompts=make_prompts(body, q, "ranking", example),
        expected={"ranking": expected},
        derivation=derivation,
        misconception_answers={k: ({"ranking": v} if v else None) for k, v in variants_for_group(g).items()},
        data=dict(teams=teams, matches=[m.__dict__ for m in matches],
                  discipline=[list(d) for d in g.discipline], fifa_rankings=g.fifa_rankings,
                  table={t: st[t] for t in teams}, conduct=conduct_scores(g)),
    ))
    return g


# ======================================================================
# Baseline controls: every plausible rule order gives the same answer.
# ======================================================================

group_task(
    "B1", "baseline", "easy", "A", ["Aldoria", "Brevonia", "Calvessa", "Dorvania"],
    [M(1, "Calvessa", 2, 0, "Dorvania"), M(1, "Aldoria", 1, 1, "Brevonia"),
     M(2, "Brevonia", 3, 1, "Dorvania"), M(2, "Aldoria", 0, 2, "Calvessa"),
     M(3, "Dorvania", 1, 0, "Aldoria"), M(3, "Brevonia", 1, 1, "Calvessa")],
    ["Calvessa", "Brevonia", "Dorvania", "Aldoria"],
    "Points: Calvessa 7, Brevonia 5, Dorvania 3, Aldoria 1. No ties, so no tie-breakers apply.",
    "Control: plain table computation, no ties.")

group_task(
    "B2", "baseline", "easy", "B", ["Estmark", "Feyland", "Galicor", "Harnland"],
    [M(1, "Estmark", 1, 1, "Feyland"), M(1, "Galicor", 1, 1, "Harnland"),
     M(2, "Estmark", 2, 0, "Galicor"), M(2, "Harnland", 0, 2, "Feyland"),
     M(3, "Harnland", 0, 1, "Estmark"), M(3, "Feyland", 0, 3, "Galicor")],
    ["Estmark", "Galicor", "Feyland", "Harnland"],
    "Points: Estmark 7, Galicor 4, Feyland 4, Harnland 1. Galicor and Feyland: Galicor won the "
    "head-to-head 3-0 (criterion a) and also has the better overall GD (+1 vs -1), so every rule "
    "order puts Galicor 2nd.",
    "Control: two-team tie where head-to-head and overall goal difference agree.")

group_task(
    "B3", "baseline", "easy", "C", ["Istrova", "Jorvania", "Lumeria", "Montrevia"],
    [M(1, "Istrova", 1, 1, "Jorvania"), M(1, "Lumeria", 2, 0, "Montrevia"),
     M(2, "Istrova", 0, 1, "Lumeria"), M(2, "Montrevia", 0, 3, "Jorvania"),
     M(3, "Montrevia", 1, 2, "Istrova"), M(3, "Jorvania", 0, 2, "Lumeria")],
    ["Lumeria", "Jorvania", "Istrova", "Montrevia"],
    "Points: Lumeria 9, Istrova 4, Jorvania 4, Montrevia 0. Istrova and Jorvania drew 1-1, so "
    "criteria a)-c) are level (1 pt, GD 0, 1 goal each). Step 2: d) overall GD Jorvania +1, "
    "Istrova 0, so Jorvania is 2nd.",
    "Control: drawn head-to-head falls through to overall goal difference.")

# ======================================================================
# Head-to-head vs overall goal difference (changed for 2026).
# ======================================================================

group_task(
    "H1", "h2h_vs_overall", "medium", "D", ["Norrland", "Ostravia", "Pellaria", "Quorvia"],
    [M(1, "Ostravia", 1, 0, "Pellaria"), M(1, "Norrland", 1, 1, "Quorvia"),
     M(2, "Norrland", 3, 0, "Ostravia"), M(2, "Pellaria", 3, 0, "Quorvia"),
     M(3, "Pellaria", 4, 0, "Norrland"), M(3, "Quorvia", 0, 1, "Ostravia")],
    ["Ostravia", "Pellaria", "Norrland", "Quorvia"],
    "Points: Ostravia 6, Pellaria 6, Norrland 4, Quorvia 1. Ostravia beat Pellaria 1-0, so "
    "criterion a) (head-to-head points 3 v 0) ranks Ostravia 1st, although Pellaria's overall GD "
    "(+6) is far better than Ostravia's (-1). Under the pre-2026 order Pellaria would top the group.",
    "Head-to-head result must outrank a much better overall goal difference.")

group_task(
    "H2", "h2h_vs_overall", "medium", "E", ["Rusmark", "Sardana", "Tavoria", "Ulmeria"],
    [M(1, "Sardana", 2, 1, "Tavoria"), M(1, "Rusmark", 2, 0, "Ulmeria"),
     M(2, "Rusmark", 4, 0, "Sardana"), M(2, "Tavoria", 5, 0, "Ulmeria"),
     M(3, "Ulmeria", 0, 0, "Sardana"), M(3, "Tavoria", 1, 1, "Rusmark")],
    ["Rusmark", "Sardana", "Tavoria", "Ulmeria"],
    "Points: Rusmark 7, Sardana 4, Tavoria 4, Ulmeria 1. Sardana beat Tavoria 2-1, so a) puts "
    "Sardana 2nd (automatic qualification) and Tavoria 3rd, despite overall GD Tavoria +4 vs "
    "Sardana -3.",
    "Same rule where the tie decides automatic qualification versus the third-place race.")

group_task(
    "H3", "h2h_vs_overall", "medium", "F", ["Valdris", "Wexland", "Yarrow", "Zentha"],
    [M(1, "Valdris", 1, 0, "Wexland"), M(1, "Yarrow", 0, 1, "Zentha"),
     M(2, "Yarrow", 1, 0, "Valdris"), M(2, "Wexland", 5, 0, "Zentha"),
     M(3, "Valdris", 1, 0, "Zentha"), M(3, "Wexland", 2, 0, "Yarrow")],
    ["Valdris", "Wexland", "Zentha", "Yarrow"],
    "Points: Valdris 6, Wexland 6, Yarrow 3, Zentha 3. Two separate two-team ties. Top: Valdris "
    "beat Wexland 1-0, so Valdris 1st (overall GD Wexland +6, Valdris +1 is irrelevant). Bottom: "
    "Zentha beat Yarrow 1-0, so Zentha 3rd (overall GD Yarrow -2, Zentha -5 is irrelevant).",
    "Two independent ties in one group, both reversed relative to overall goal difference.")

# ======================================================================
# Three-way ties: mini-table and step-2 re-application.
# ======================================================================

group_task(
    "T1", "three_way", "medium", "G", ["Galicor", "Harnland", "Istrova", "Jorvania"],
    [M(1, "Galicor", 3, 0, "Harnland"), M(1, "Istrova", 1, 0, "Jorvania"),
     M(2, "Harnland", 1, 0, "Istrova"), M(2, "Jorvania", 0, 1, "Galicor"),
     M(3, "Istrova", 2, 1, "Galicor"), M(3, "Harnland", 5, 0, "Jorvania")],
    ["Galicor", "Istrova", "Harnland", "Jorvania"],
    "Galicor, Harnland and Istrova all have 6 points (cyclic results). Mini-table of their three "
    "matches: points 3 each; GD Galicor +2 (3-0, 1-2), Istrova 0 (0-1, 2-1), Harnland -2 (0-3, 1-0). "
    "So Galicor, Istrova, Harnland. Overall GD (Galicor +3, Harnland +3, Istrova +1) is not used.",
    "Three-way tie resolved by the head-to-head mini-table, not by overall goal difference.")

group_task(
    "T2", "three_way", "hard", "H", ["Lumeria", "Norrland", "Ostravia", "Pellaria"],
    [M(1, "Lumeria", 3, 0, "Norrland"), M(1, "Pellaria", 4, 0, "Ostravia"),
     M(2, "Norrland", 2, 0, "Pellaria"), M(2, "Ostravia", 0, 1, "Lumeria"),
     M(3, "Pellaria", 2, 1, "Lumeria"), M(3, "Norrland", 1, 0, "Ostravia")],
    ["Lumeria", "Norrland", "Pellaria", "Ostravia"],
    "Lumeria, Norrland and Pellaria have 6 points. Mini-table: 3 points each; GD Lumeria +2 "
    "(3-0, 1-2), Norrland -1 (0-3, 2-0), Pellaria -1 (0-2, 2-1); goals Norrland 2, Pellaria 2. "
    "Lumeria is 1st; Norrland and Pellaria are still level after a)-c). Step 2: re-apply a)-c) to "
    "the match between Norrland and Pellaria only: Norrland won 2-0, so Norrland 2nd, Pellaria 3rd. "
    "Going straight to overall GD (Pellaria +3, Norrland 0) is wrong.",
    "Step-2 re-application of head-to-head to the remaining two teams.")

group_task(
    "T3", "three_way", "hard", "I", ["Quorvia", "Rusmark", "Sardana", "Tavoria"],
    [M(1, "Quorvia", 1, 0, "Rusmark"), M(1, "Sardana", 0, 0, "Tavoria"),
     M(2, "Quorvia", 3, 0, "Sardana"), M(2, "Rusmark", 0, 0, "Tavoria"),
     M(3, "Tavoria", 0, 2, "Quorvia"), M(3, "Rusmark", 1, 1, "Sardana")],
    ["Quorvia", "Rusmark", "Sardana", "Tavoria"],
    "Quorvia 9; Rusmark, Sardana, Tavoria 2 each (all three matches among them drawn). Mini-table: "
    "2 pts and GD 0 each; goals scored Rusmark 1, Sardana 1, Tavoria 0, so Tavoria is 4th. Step 2: "
    "Rusmark v Sardana alone was 1-1, still level, so d) overall GD: Rusmark -1, Sardana -3. Final: "
    "Quorvia, Rusmark, Sardana, Tavoria. Overall GD alone (Tavoria -2) would wrongly put Tavoria 3rd.",
    "Mini-table goals scored separates a team that overall goal difference would favour.")

# ======================================================================
# Step 2 'does not restart' clause.
# ======================================================================

group_task(
    "N1", "no_restart", "hard", "J", ["Ulmeria", "Valdris", "Wexland", "Yarrow"],
    [M(1, "Ulmeria", 1, 0, "Valdris"), M(1, "Wexland", 3, 2, "Yarrow"),
     M(2, "Valdris", 1, 0, "Wexland"), M(2, "Yarrow", 0, 3, "Ulmeria"),
     M(3, "Wexland", 1, 0, "Ulmeria"), M(3, "Valdris", 1, 0, "Yarrow")],
    ["Ulmeria", "Wexland", "Valdris", "Yarrow"],
    "Ulmeria, Valdris, Wexland have 6 points; their mutual matches were all 1-0 wins in a cycle, so "
    "the mini-table is identical (3 pts, GD 0, 1 goal) and re-application changes nothing. Step 2 "
    "d) overall GD: Ulmeria +3, Valdris +1, Wexland +1, so Ulmeria 1st. The step does not restart: "
    "e) overall goals Wexland 4 v Valdris 2, so Wexland 2nd even though Valdris beat Wexland 1-0.",
    "Once in step 2, the remaining teams are NOT re-compared head-to-head.")

group_task(
    "N2", "no_restart", "hard", "K", ["Calvessa", "Estmark", "Harnland", "Montrevia"],
    [M(1, "Calvessa", 1, 0, "Estmark"), M(1, "Harnland", 3, 1, "Montrevia"),
     M(2, "Estmark", 1, 0, "Harnland"), M(2, "Montrevia", 0, 2, "Calvessa"),
     M(3, "Harnland", 1, 0, "Calvessa"), M(3, "Estmark", 3, 1, "Montrevia")],
    ["Harnland", "Estmark", "Calvessa", "Montrevia"],
    "Calvessa, Estmark, Harnland have 6 points; mutual matches a cycle of 1-0 wins, so a)-c) cannot "
    "separate them. d) overall GD +2 for all three. e) goals: Estmark 4, Harnland 4, Calvessa 3, so "
    "Calvessa 3rd. No restart, so f) conduct for Estmark v Harnland: Estmark -3 (player sent off for "
    "two yellows) -1 (head coach yellow) = -4; Harnland three single yellows = -3. Harnland 1st, "
    "Estmark 2nd, despite Estmark's 1-0 head-to-head win and better world ranking.",
    "No-restart clause combined with the conduct score, including a team-official card.",
    discipline=[(2, "Estmark", "#6", ["Y", "Y"]), (3, "Estmark", "head coach", ["Y"]),
                (1, "Harnland", "#4", ["Y"]), (2, "Harnland", "#9", ["Y"]), (3, "Harnland", "#15", ["Y"]),
                (1, "Calvessa", "#8", ["Y"]), (2, "Montrevia", "#3", ["Y"]), (2, "Montrevia", "#14", ["Y"])],
    rankings=[{"Estmark": 20, "Harnland": 35, "Calvessa": 41, "Montrevia": 77}])

# ======================================================================
# Team conduct score and world ranking.
# ======================================================================

group_task(
    "C1", "conduct", "medium", "L", ["Ostravia", "Brevonia", "Dorvania", "Feyland"],
    [M(1, "Ostravia", 2, 0, "Brevonia"), M(1, "Dorvania", 2, 1, "Feyland"),
     M(2, "Ostravia", 2, 0, "Dorvania"), M(2, "Brevonia", 2, 1, "Feyland"),
     M(3, "Feyland", 0, 1, "Ostravia"), M(3, "Brevonia", 1, 1, "Dorvania")],
    ["Ostravia", "Dorvania", "Brevonia", "Feyland"],
    "Brevonia and Dorvania: 4 pts, drew 1-1, identical overall record (GF 3, GA 4). Conduct: "
    "Brevonia four single yellows to different players = -4; Dorvania one player shown two "
    "yellows in one match = indirect red = -3 (only one deduction per person per match, not "
    "-1-1-3). Dorvania -3 > Brevonia -4, so Dorvania 2nd. The world ranking (Brevonia better) is "
    "not reached.",
    "Conduct score where a second yellow counts as -3, not as separate cards.",
    discipline=[(1, "Brevonia", "#3", ["Y"]), (2, "Brevonia", "#7", ["Y"]), (2, "Brevonia", "#11", ["Y"]),
                (3, "Brevonia", "#16", ["Y"]), (2, "Dorvania", "#5", ["Y", "Y"]),
                (1, "Ostravia", "#10", ["Y"]), (3, "Feyland", "#2", ["Y"]), (3, "Feyland", "#19", ["Y"])],
    rankings=[{"Ostravia": 9, "Brevonia": 24, "Dorvania": 47, "Feyland": 63}])

group_task(
    "C2", "conduct", "hard", "A", ["Galicor", "Istrova", "Lumeria", "Norrland"],
    [M(1, "Galicor", 1, 0, "Istrova"), M(1, "Lumeria", 2, 0, "Norrland"),
     M(2, "Galicor", 1, 0, "Lumeria"), M(2, "Istrova", 2, 0, "Norrland"),
     M(3, "Norrland", 1, 1, "Galicor"), M(3, "Istrova", 0, 0, "Lumeria")],
    ["Galicor", "Lumeria", "Istrova", "Norrland"],
    "Istrova and Lumeria: 4 pts, drew 0-0, identical overall record (GF 2, GA 1). Conduct: "
    "Istrova head coach yellow (-1) + player #10 one yellow on matchday 1 and one on matchday 3 "
    "(separate matches, so -1 and -1, not an indirect red) = -3. Lumeria player #5 two yellows in "
    "one match = -3. Level, so g) world ranking: Lumeria 22nd, Istrova 40th, Lumeria 2nd.",
    "Conduct score with an official's card and yellows in different matches, ending in a tie.",
    discipline=[(3, "Istrova", "head coach", ["Y"]), (1, "Istrova", "#10", ["Y"]), (3, "Istrova", "#10", ["Y"]),
                (2, "Lumeria", "#5", ["Y", "Y"]), (1, "Galicor", "#6", ["Y"]), (3, "Norrland", "#8", ["Y"])],
    rankings=[{"Galicor": 12, "Lumeria": 22, "Istrova": 40, "Norrland": 71}])

group_task(
    "R1", "world_ranking", "medium", "B", ["Pellaria", "Quorvia", "Rusmark", "Sardana"],
    [M(1, "Pellaria", 2, 1, "Quorvia"), M(1, "Rusmark", 1, 0, "Sardana"),
     M(2, "Pellaria", 2, 1, "Rusmark"), M(2, "Quorvia", 1, 0, "Sardana"),
     M(3, "Sardana", 0, 3, "Pellaria"), M(3, "Quorvia", 2, 2, "Rusmark")],
    ["Pellaria", "Rusmark", "Quorvia", "Sardana"],
    "Quorvia and Rusmark: 4 pts, drew 2-2, identical overall record (GF 4, GA 4), conduct -2 each "
    "(Rusmark's includes an assistant coach yellow). Step 3 g): joint 27th in the most recent "
    "ranking, so h): previous edition Rusmark 25th, Quorvia 28th. Rusmark 2nd. Drawing of lots "
    "(the pre-2026 last resort) no longer exists.",
    "Reaching step 3 and falling back to the preceding world-ranking edition.",
    discipline=[(1, "Quorvia", "#3", ["Y"]), (3, "Quorvia", "#8", ["Y"]),
                (2, "Rusmark", "#11", ["Y"]), (2, "Rusmark", "assistant coach", ["Y"]),
                (1, "Pellaria", "#7", ["Y"]), (2, "Sardana", "#4", ["Y"]), (3, "Sardana", "#20", ["Y"])],
    rankings=[{"Pellaria": 14, "Quorvia": 27, "Rusmark": 27, "Sardana": 66},
              {"Pellaria": 15, "Rusmark": 25, "Quorvia": 28, "Sardana": 64}])

# ======================================================================
# Four-way tie (all teams level on points).
# ======================================================================

group_task(
    "F1", "four_way", "hard", "C", ["Aldoria", "Jorvania", "Kestria", "Valdris"],
    [M(1, "Aldoria", 2, 2, "Jorvania"), M(1, "Kestria", 1, 1, "Valdris"),
     M(2, "Aldoria", 1, 0, "Kestria"), M(2, "Valdris", 0, 1, "Jorvania"),
     M(3, "Kestria", 2, 1, "Jorvania"), M(3, "Valdris", 2, 0, "Aldoria")],
    ["Valdris", "Jorvania", "Kestria", "Aldoria"],
    "All four teams have 4 points, so the head-to-head table is the whole group. a) level; b) GD "
    "Valdris +1, Jorvania 0, Kestria 0, Aldoria -1; c) goals Jorvania 4, Kestria 3, so Jorvania "
    "above Kestria. Every team is now separated, so nothing is re-applied: Kestria's 2-1 win over "
    "Jorvania is never used.",
    "Four-way tie where criterion c) separates two teams before any pairwise comparison.")

# ======================================================================
# Ranking of the third-placed teams.
# ======================================================================


def thirds_task(tid: str, difficulty: str, rows: list[dict], expected_top8: list[str], derivation: str,
                tests: str, group_l: Group | None = None) -> None:
    full = rank_thirds(rows)
    assert full[:8] == expected_top8, f"{tid}: engine {full[:8]} != hand-derived {expected_top8}"
    lines = []
    if group_l is not None:
        lines += [render_group_block(group_l, with_ranking=False), "",
                  "FIFA/Coca-Cola Men's World Ranking, most recent published edition: " +
                  ", ".join(f"{t} {ordinal(group_l.fifa_rankings[0][t])}" for t in group_l.teams), "",
                  "Third-placed teams in Groups A to K (all group-stage figures; conduct = team conduct score):"]
    else:
        lines.append("Third-placed teams after the group stage (all group-stage figures; conduct = team conduct score):")
    lines.append("Group | Team | Pts | GD | GF | Conduct | World ranking")
    for r in rows:
        if group_l is not None and r["group"] == "L":
            continue
        lines.append(f"{r['group']} | {r['team']} | {r['pts']} | {r['gd']:+d} | {r['gf']} | {r['conduct']} | "
                     f"{ordinal(r['fifa_rankings'][0])}")
    body = "\n".join(lines)
    if group_l is not None:
        q = ("All group matches have been played. Determine the third-placed team of Group L, then rank all "
             "twelve third-placed teams and list the eight that qualify for the round of 32, in rank order "
             "(best first).")
    else:
        q = ("Rank the twelve third-placed teams under the FIFA World Cup 26 regulations and list the eight "
             "that qualify for the round of 32, in rank order (best first).")
    example = json.dumps([f"<{ordinal(i)} best third>" for i in range(1, 9)])
    variants = {}
    for name, fn in THIRD_VARIANTS.items():
        try:
            variants[name] = {"ranking": fn(rows)[:8]}
        except UnresolvedTie:
            variants[name] = None
    if group_l is not None:
        variants["legacy_overall_first"] = {"ranking": _thirds_with_l3(rows, group_l, rank_legacy_overall_first)[:8]}
    TASKS.append(dict(
        id=tid, type="thirds_ranking", category="third_place", difficulty=difficulty,
        what_it_tests=tests, answer_key="ranking",
        prompts=make_prompts(body, q, "ranking", example),
        expected={"ranking": expected_top8},
        derivation=derivation,
        misconception_answers=variants,
        data=dict(thirds=rows, full_ranking=full, group_l_teams=group_l.teams if group_l else [],
                  group_l_matches=[m.__dict__ for m in group_l.matches] if group_l else []),
    ))


def third_row(g: Group, ranker, group_letter: str = "L") -> dict:
    team = ranker(g)[2]
    st = table(g.teams, g.matches)[team]
    return dict(group=group_letter, team=team, pts=st["pts"], gd=st["gd"], gf=st["gf"],
                conduct=conduct_scores(g)[team], fifa_rankings=[g.fifa_rankings[0][team]])


def _thirds_with_l3(rows: list[dict], g: Group, ranker) -> list[str]:
    return rank_thirds([r for r in rows if r["group"] != "L"] + [third_row(g, ranker)])


def row(group, team, pts, gd, gf, conduct, rank):
    return dict(group=group, team=team, pts=pts, gd=gd, gf=gf, conduct=conduct, fifa_rankings=[rank])


thirds_task(
    "P1", "medium",
    [row("A", "Aldoria", 6, 3, 6, -4, 34), row("B", "Brevonia", 4, 1, 4, -2, 51),
     row("C", "Calvessa", 4, 0, 5, -6, 22), row("D", "Dorvania", 4, 0, 5, -3, 60),
     row("E", "Estmark", 4, 0, 3, -1, 18), row("F", "Feyland", 3, 1, 4, -5, 40),
     row("G", "Galicor", 3, 0, 4, -2, 29), row("H", "Harnland", 3, 0, 4, -2, 44),
     row("I", "Istrova", 3, 0, 2, 0, 15), row("J", "Jorvania", 3, -1, 5, -1, 33),
     row("K", "Kestria", 3, -2, 3, -3, 27), row("L", "Lumeria", 1, -3, 2, -2, 55)],
    ["Aldoria", "Brevonia", "Dorvania", "Calvessa", "Estmark", "Feyland", "Galicor", "Harnland"],
    "Order: points, GD, goals, conduct, world ranking (no head-to-head between different groups). "
    "6 pts: Aldoria. 4 pts: Brevonia (+1); Dorvania and Calvessa (0, 5 goals) split by conduct "
    "-3 > -6; Estmark (0, 3 goals). 3 pts: Feyland (+1); Galicor and Harnland (0, 4 goals, -2) split by "
    "world ranking 29th v 44th; then Istrova (0, 2 goals), Jorvania (-1), Kestria (-2). Top eight end "
    "with Harnland; Istrova (9th) misses out despite the best conduct and ranking.",
    "Best-third ranking: GD before goals, conduct direction, ranking as last resort.")

GL = Group("L", ["Quorvia", "Rusmark", "Sardana", "Tavoria"],
           [M(1, "Quorvia", 1, 0, "Rusmark"), M(1, "Sardana", 0, 0, "Tavoria"),
            M(2, "Quorvia", 3, 0, "Sardana"), M(2, "Rusmark", 0, 0, "Tavoria"),
            M(3, "Tavoria", 0, 2, "Quorvia"), M(3, "Rusmark", 1, 1, "Sardana")],
           [(1, "Rusmark", "#4", ["Y"]), (3, "Rusmark", "#13", ["Y"]), (2, "Sardana", "#7", ["Y"]),
            (1, "Tavoria", "#2", ["Y"]), (2, "Tavoria", "#6", ["Y"]), (3, "Tavoria", "#18", ["Y"])],
           [{"Quorvia": 11, "Rusmark": 38, "Sardana": 52, "Tavoria": 46}])
assert rank_fwc26(GL) == ["Quorvia", "Rusmark", "Sardana", "Tavoria"]
assert rank_legacy_overall_first(GL)[2] == "Tavoria"

thirds_task(
    "P2", "hard",
    [row("A", "Aldoria", 4, 2, 4, -1, 30), row("B", "Brevonia", 4, 1, 3, -2, 41),
     row("C", "Calvessa", 4, 0, 4, -3, 25), row("D", "Dorvania", 3, 1, 3, -1, 50),
     row("E", "Estmark", 3, 0, 4, -2, 19), row("F", "Feyland", 3, 0, 3, -4, 37),
     row("G", "Galicor", 3, -1, 2, 0, 58), row("H", "Harnland", 1, -4, 1, -6, 70),
     row("I", "Istrova", 2, -4, 2, -2, 45), row("J", "Jorvania", 1, -2, 2, -1, 66),
     row("K", "Kestria", 2, -3, 2, -2, 48), third_row(GL, rank_fwc26)],
    ["Aldoria", "Brevonia", "Calvessa", "Dorvania", "Estmark", "Feyland", "Galicor", "Kestria"],
    "Group L: Quorvia 9; Rusmark, Sardana, Tavoria 2 each, all mutual matches drawn. Mini-table goals: "
    "Rusmark 1, Sardana 1, Tavoria 0, so Tavoria 4th; Rusmark v Sardana 1-1, then overall GD Rusmark -1 "
    "> Sardana -3. Third place: Sardana (2 pts, GD -3, 1 goal). Thirds: seven teams with 3+ points; "
    "8th place among the 2-point teams: Kestria (-3, 2 goals) > Sardana (-3, 1 goal) > Istrova (-4). "
    "Kestria takes the last spot. (Pre-2026 order would make Tavoria third in L, and Tavoria's -2 GD "
    "would take the 8th spot instead.)",
    "Two-stage reasoning: a group tie-break feeds the cross-group third-place cut-off.",
    group_l=GL)

# ======================================================================
# Final-matchday scenarios (which results let a team through).
# ======================================================================


def scenario_task(tid: str, difficulty: str, g0: Group, fixed: Match, target: str, other: tuple[str, str],
                  options: list[tuple[int, int]], expected: list[str], derivation: str, tests: str) -> None:
    letters = "abcdefgh"
    good, legacy_good = [], []
    for letter, (hg, ag) in zip(letters, options):
        g = Group(g0.name, g0.teams, g0.matches + [fixed, M(3, other[0], hg, ag, other[1])],
                  g0.discipline, g0.fifa_rankings)
        if target in rank_fwc26(g)[:2]:
            good.append(letter)
        if target in rank_legacy_overall_first(g)[:2]:
            legacy_good.append(letter)
    assert good == expected, f"{tid}: engine {good} != hand-derived {expected}"
    body = render_group_block(g0).replace("Results:", "Results so far (matchdays 1 and 2):")
    opts = "\n".join(f"({l}) {other[0]} {h}-{a} {other[1]}" for l, (h, a) in zip(letters, options))
    q = (f"On matchday 3, {fixed.home} and {fixed.away} have finished {fixed.hg}-{fixed.ag}. "
         f"{other[0]} v {other[1]} is still to be completed, and no further cards will be shown. "
         f"For which of the following final scores of {other[0]} v {other[1]} does {target} finish in the "
         f"top two of Group {g0.name}?\n{opts}")
    TASKS.append(dict(
        id=tid, type="scenario", category="scenario", difficulty=difficulty,
        what_it_tests=tests, answer_key="options",
        prompts=make_prompts(body, q, "options", '["<letter>", "..."]'),
        expected={"options": expected},
        derivation=derivation,
        misconception_answers={"legacy_overall_first": {"options": legacy_good}},
        data=dict(teams=g0.teams, matches=[m.__dict__ for m in g0.matches], fixed=fixed.__dict__, other=list(other),
                  target=target, options=options, discipline=[list(d) for d in g0.discipline],
                  fifa_rankings=g0.fifa_rankings),
    ))


S1 = Group("D", ["Brevonia", "Estmark", "Montrevia", "Zentha"],
           [M(1, "Brevonia", 2, 0, "Montrevia"), M(1, "Estmark", 1, 0, "Zentha"),
            M(2, "Montrevia", 3, 0, "Zentha"), M(2, "Brevonia", 0, 1, "Estmark")])
S1.discipline = background_discipline("S1", S1.teams, matchdays=2)
S1.fifa_rankings = background_rankings("S1", S1.teams)
scenario_task(
    "S1", "hard", S1, M(3, "Estmark", 2, 0, "Montrevia"), "Montrevia", ("Brevonia", "Zentha"),
    [(0, 0), (1, 1), (0, 1), (1, 2), (0, 3), (2, 4)],
    ["c", "e", "f"],
    "After Estmark 2-0 Montrevia: Estmark 9; Montrevia 3 (GF 3, GA 4); Brevonia 3 (GF 2, GA 1); "
    "Zentha 0. (a), (b): a draw gives Brevonia 4 points, Montrevia 3rd. If Zentha wins, Brevonia, "
    "Montrevia and Zentha all have 3 points and the mini-table of their matches decides (Brevonia 2-0 "
    "Montrevia, Montrevia 3-0 Zentha, Zentha beat Brevonia). (c) 0-1: mini GD Brevonia +1, Montrevia "
    "+1, Zentha -2; mini goals Montrevia 3, Brevonia 2: Montrevia 2nd. (d) 1-2: mini GD +1/+1/-2 and "
    "mini goals 3 v 3, still level; re-apply to Brevonia v Montrevia only: Brevonia won 2-0, Montrevia "
    "3rd. (e) 0-3: mini GD Montrevia +1, Zentha 0, Brevonia -1: Montrevia 2nd. (f) 2-4: mini GD "
    "Montrevia +1, Brevonia 0, Zentha -1: Montrevia 2nd. Answer c, e, f (the pre-2026 order gives e only).",
    "Final-day permutations decided by a three-way mini-table, including a step-2 re-application.")

S2 = Group("E", ["Feyland", "Kestria", "Ostravia", "Ulmeria"],
           [M(1, "Kestria", 1, 0, "Ostravia"), M(1, "Feyland", 2, 0, "Ulmeria"),
            M(2, "Ostravia", 2, 0, "Ulmeria"), M(2, "Feyland", 1, 1, "Kestria")])
S2.discipline = background_discipline("S2", S2.teams, matchdays=2)
S2.fifa_rankings = background_rankings("S2", S2.teams)
scenario_task(
    "S2", "medium", S2, M(3, "Feyland", 3, 0, "Kestria"), "Kestria", ("Ostravia", "Ulmeria"),
    [(2, 0), (0, 0), (3, 3), (0, 1), (1, 0)],
    ["b", "c", "d"],
    "After Feyland 3-0 Kestria: Feyland 7, Kestria finishes on 4 points (GF 2, GA 4), Ostravia 3, "
    "Ulmeria 0. An Ostravia win (a, e) takes Ostravia to 6: Kestria 3rd. A draw (b, c) leaves Ostravia "
    "level with Kestria on 4; Kestria beat Ostravia 1-0 on matchday 1, so criterion a) puts Kestria 2nd "
    "even though Ostravia's overall goal difference (+1) is better than Kestria's (-2). An Ulmeria win "
    "(d) leaves Ostravia on 3: Kestria 2nd. Answer b, c, d (the pre-2026 order gives d only).",
    "Final-day permutations where an earlier head-to-head result beats goal difference.")


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for t in TASKS:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(TASKS)} tasks to {OUT.relative_to(ROOT)}")
