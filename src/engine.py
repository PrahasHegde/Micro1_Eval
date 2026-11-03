"""Reference implementation of group-ranking rules.

`rank_fwc26` implements Article 13 of the FIFA World Cup 26 Regulations
(FWC26_regulations_EN.pdf, pp. 26-27). It is the source of truth for every
expected answer in tasks/tasks.jsonl, alongside a hand-written derivation.

The other `rank_*` functions are deliberately *wrong* procedures. Each one
models a specific misconception (e.g. using the pre-2026 order of criteria).
The grader uses them to diagnose why a model answer is wrong: if the answer
equals the output of a misconception, that misconception is the likely cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


class UnresolvedTie(Exception):
    """Raised when the criteria are exhausted and teams are still level."""


@dataclass(frozen=True)
class Match:
    home: str
    away: str
    hg: int
    ag: int
    matchday: int = 0


@dataclass
class Group:
    name: str
    teams: list[str]
    matches: list[Match]
    # (matchday, team, person, cards) where cards is a list of "Y" / "R"
    discipline: list[tuple[int, str, str, list[str]]] = field(default_factory=list)
    # FIFA/Coca-Cola World Ranking editions, most recent first: {team: position}
    fifa_rankings: list[dict[str, int]] = field(default_factory=list)


# ---------------------------------------------------------------- statistics

def table(teams: Iterable[str], matches: Iterable[Match]) -> dict[str, dict]:
    """Points / GF / GA / GD for `teams`, counting only matches among them."""
    teams = set(teams)
    st = {t: dict(pts=0, w=0, d=0, l=0, gf=0, ga=0) for t in teams}
    for m in matches:
        if m.home not in teams or m.away not in teams:
            continue
        for me, them, f, a in ((m.home, m.away, m.hg, m.ag), (m.away, m.home, m.ag, m.hg)):
            s = st[me]
            s["gf"] += f
            s["ga"] += a
            if f > a:
                s["pts"] += 3
                s["w"] += 1
            elif f == a:
                s["pts"] += 1
                s["d"] += 1
            else:
                s["l"] += 1
    for s in st.values():
        s["gd"] = s["gf"] - s["ga"]
    return st


def person_deduction(cards: list[str]) -> int:
    """Art. 13 step 2 f): only one deduction per person per match."""
    y, r = cards.count("Y"), cards.count("R")
    if r and y:
        return -5  # yellow card and direct red card
    if r:
        return -4  # direct red card
    if y >= 2:
        return -3  # indirect red card (second yellow)
    if y == 1:
        return -1
    return 0


def naive_person_deduction(cards: list[str]) -> int:
    """Misconception: every card is penalised separately (Y+Y+indirect red = -5)."""
    y, r = cards.count("Y"), cards.count("R")
    return -1 * y + (-3 if y >= 2 and not r else 0) + (-4 * r)


def conduct_scores(g: Group, per_person: Callable[[list[str]], int] = person_deduction) -> dict[str, int]:
    scores = {t: 0 for t in g.teams}
    for _md, team, _person, cards in g.discipline:
        scores[team] += per_person(cards)
    return scores


# ------------------------------------------------------------------ helpers

def partition(teams: list[str], key: Callable[[str], tuple]) -> list[list[str]]:
    """Sort teams by key (higher is better) and group equal keys together."""
    ordered = sorted(teams, key=key, reverse=True)
    blocks: list[list[str]] = []
    for t in ordered:
        if blocks and key(blocks[-1][0]) == key(t):
            blocks[-1].append(t)
        else:
            blocks.append([t])
    return blocks


def h2h_key(g: Group, teams: list[str]) -> Callable[[str], tuple]:
    mini = table(teams, g.matches)
    return lambda t: (mini[t]["pts"], mini[t]["gd"], mini[t]["gf"])


def ranking_keys(g: Group) -> list[Callable[[str], tuple]]:
    # Lower world-ranking position is better, hence the negation.
    return [(lambda t, ed=ed: (-ed[t],)) for ed in g.fifa_rankings]


def sequential(teams: list[str], keys: list[Callable[[str], tuple]]) -> list[str]:
    """Apply criteria in order without restarting (Art. 13 step 2 / step 3)."""
    if len(teams) == 1:
        return teams
    if not keys:
        raise UnresolvedTie(teams)
    out: list[str] = []
    for block in partition(teams, keys[0]):
        out += sequential(block, keys[1:])
    return out


def _rank_by_points(g: Group, resolve: Callable[[list[str]], list[str]]) -> list[str]:
    st = table(g.teams, g.matches)
    out: list[str] = []
    for block in partition(g.teams, lambda t: (st[t]["pts"],)):
        out += block if len(block) == 1 else resolve(block)
    return out


# ------------------------------------------------------------ FWC26 (correct)

def rank_fwc26(g: Group, *, conduct: Callable[[list[str]], int] = person_deduction,
               reapply: bool = True, restart: bool = False) -> list[str]:
    """FIFA World Cup 26 Regulations, Article 13.

    Step 1: a)-c) head-to-head points, GD, goals among the teams concerned.
    Step 2: if only some teams separate, re-apply a)-c) to the matches between
            the remaining teams only; if that decides nothing, apply d) overall
            GD, e) overall goals, f) team conduct score, in order, WITHOUT
            restarting for the teams left after a criterion separates others.
    Step 3: g)/h) FIFA/Coca-Cola World Ranking, most recent edition first.

    `conduct`, `reapply` and `restart` exist only to build misconception variants.
    """
    st = table(g.teams, g.matches)
    cs = conduct_scores(g, conduct)
    overall = [lambda t: (st[t]["gd"],), lambda t: (st[t]["gf"],), lambda t: (cs[t],)]
    overall += ranking_keys(g)

    def step2(teams: list[str]) -> list[str]:
        if not restart:
            return sequential(teams, overall)
        # Misconception: go back to head-to-head after each overall criterion.
        for i, key in enumerate(overall):
            blocks = partition(teams, key)
            if len(blocks) > 1:
                out: list[str] = []
                for b in blocks:
                    out += b if len(b) == 1 else resolve(b)
                return out
        raise UnresolvedTie(teams)

    def resolve(teams: list[str]) -> list[str]:
        blocks = partition(teams, h2h_key(g, teams))
        if len(blocks) == 1:
            return step2(teams)
        out: list[str] = []
        for b in blocks:
            if len(b) == 1:
                out += b
            else:
                out += resolve(b) if reapply else step2(b)
        return out

    return _rank_by_points(g, resolve)


# -------------------------------------------------------- misconceptions

def rank_legacy_overall_first(g: Group) -> list[str]:
    """Pre-2026 FIFA order (2018/2022 World Cups): overall GD and goals first,
    then head-to-head among the teams concerned, then fair play, then lots
    (modelled here with the world ranking so that it always resolves)."""
    st = table(g.teams, g.matches)
    cs = conduct_scores(g)

    def resolve(teams: list[str]) -> list[str]:
        out: list[str] = []
        for block in partition(teams, lambda t: (st[t]["gd"], st[t]["gf"])):
            if len(block) == 1:
                out += block
            else:
                out += sequential(block, [h2h_key(g, block), lambda t: (cs[t],)] + ranking_keys(g))
        return out

    return _rank_by_points(g, resolve)


def rank_no_reapply(g: Group) -> list[str]:
    """Head-to-head once, then straight to overall criteria for any sub-tie."""
    return rank_fwc26(g, reapply=False)


def rank_restart(g: Group) -> list[str]:
    """Returns to head-to-head after an overall criterion separates a team."""
    return rank_fwc26(g, restart=True)


def rank_naive_conduct(g: Group) -> list[str]:
    """Counts each card separately instead of one deduction per person per match."""
    return rank_fwc26(g, conduct=naive_person_deduction)


def rank_ignore_officials(g: Group) -> list[str]:
    """Leaves cards shown to team officials out of the conduct score."""
    g2 = Group(g.name, g.teams, g.matches,
               [d for d in g.discipline if not d[2].lower().startswith(("coach", "head coach", "official", "assistant"))],
               g.fifa_rankings)
    return rank_fwc26(g2)


GROUP_VARIANTS: dict[str, Callable[[Group], list[str]]] = {
    "legacy_overall_first": rank_legacy_overall_first,
    "no_reapply": rank_no_reapply,
    "restart_h2h": rank_restart,
    "naive_conduct": rank_naive_conduct,
    "ignore_officials": rank_ignore_officials,
}


# ------------------------------------------------------- third-placed teams

def rank_thirds(rows: list[dict]) -> list[str]:
    """Art. 13 'eight best-ranked third-placed teams': points, GD, goals,
    conduct score, then world-ranking editions. No head-to-head."""
    by = {r["team"]: r for r in rows}
    n_editions = len(rows[0]["fifa_rankings"])
    keys = [lambda t: (by[t]["pts"],), lambda t: (by[t]["gd"],), lambda t: (by[t]["gf"],),
            lambda t: (by[t]["conduct"],)]
    keys += [(lambda t, i=i: (-by[t]["fifa_rankings"][i],)) for i in range(n_editions)]
    return sequential([r["team"] for r in rows], keys)


def rank_thirds_gf_before_gd(rows: list[dict]) -> list[str]:
    by = {r["team"]: r for r in rows}
    keys = [lambda t: (by[t]["pts"],), lambda t: (by[t]["gf"],), lambda t: (by[t]["gd"],),
            lambda t: (by[t]["conduct"],), lambda t: (-by[t]["fifa_rankings"][0],)]
    return sequential([r["team"] for r in rows], keys)


def rank_thirds_conduct_inverted(rows: list[dict]) -> list[str]:
    """Treats the conduct score as 'fewer points is better' (most negative wins)."""
    flipped = [dict(r, conduct=-r["conduct"]) for r in rows]
    return rank_thirds(flipped)


THIRD_VARIANTS = {
    "gf_before_gd": rank_thirds_gf_before_gd,
    "conduct_inverted": rank_thirds_conduct_inverted,
}
