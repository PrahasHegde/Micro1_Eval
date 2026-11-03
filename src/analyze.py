"""Aggregates results/scores.csv into results/summary.md and results/figures/*.png.

    python src/analyze.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "results" / "scores.csv"
SUMMARY = ROOT / "results" / "summary.md"
FIG_DIR = ROOT / "results" / "figures"
TASKS = ROOT / "tasks" / "tasks.jsonl"

CATEGORY_ORDER = ["baseline", "h2h_vs_overall", "three_way", "no_restart", "conduct", "world_ranking",
                  "four_way", "third_place", "scenario"]
CONDITION_LABEL = {"closed_book": "Closed book", "open_book": "Open book (Art. 13 supplied)"}

# Reference palette (dataviz skill): categorical slots 1-2, blue sequential ramp, text tokens.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"closed_book": "#2a78d6", "open_book": "#eb6834"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def load() -> list[dict]:
    with SCORES.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["correct"] = int(r["correct"])
        r["parse_ok"] = int(r["parse_ok"])
        r["partial"] = float(r["partial"])
    return [r for r in rows if not r["error"]]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_overall(models: list[str], conditions: list[str], acc: dict, n: dict) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(models) + 2), 4), dpi=160, facecolor=SURFACE)
    style_axes(ax)
    width = 0.8 / len(conditions)
    for i, c in enumerate(conditions):
        xs = [m + (i - (len(conditions) - 1) / 2) * width for m in range(len(models))]
        vals = [acc.get((mod, c), 0) for mod in models]
        ax.bar(xs, vals, width=width - 0.03, color=SERIES.get(c, "#6250d6"), label=CONDITION_LABEL.get(c, c),
               edgecolor=SURFACE, linewidth=2)
        for x, v, mod in zip(xs, vals, models):
            ax.text(x, v + 0.015, pct(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(models)), models, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Tasks fully correct", color=INK_2)
    ax.set_title("Exact-answer accuracy by model and condition", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "accuracy_by_model.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def chart_heatmap(models: list[str], cats: list[str], cat_acc: dict, condition: str) -> Path:
    fig, ax = plt.subplots(figsize=(1.1 * len(cats) + 2.5, 0.55 * len(models) + 1.6), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    data = [[cat_acc.get((m, condition, c), float("nan")) for c in cats] for m in models]
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            if not math.isnan(v):
                ax.text(j, i, pct(v), ha="center", va="center", fontsize=8, color=INK if v < 0.55 else "#ffffff")
    ax.set_xticks(range(len(cats)), [c.replace("_", " ") for c in cats], rotation=30, ha="right", color=INK)
    ax.set_yticks(range(len(models)), models, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cats))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_title(f"Accuracy by task category ({CONDITION_LABEL.get(condition, condition).lower()})",
                 color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"category_heatmap_{condition}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit("no graded rows; run src/grade.py first")
    with TASKS.open(encoding="utf-8") as f:
        tasks = {t["id"]: t for t in map(json.loads, f)}
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    models = sorted({r["model"] for r in rows})
    conditions = [c for c in ("closed_book", "open_book") if any(r["condition"] == c for r in rows)]
    cats = [c for c in CATEGORY_ORDER if any(r["category"] == c for r in rows)]
    task_ids = list(tasks)

    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["condition"])].append(r)
        by[(r["model"], r["condition"], r["category"])].append(r)
        by[(r["model"], r["condition"], "task", r["task_id"])].append(r)

    acc = {k: sum(r["correct"] for r in v) / len(v) for k, v in by.items() if len(k) == 2}
    n = {k: len(v) for k, v in by.items() if len(k) == 2}
    cat_acc = {k: sum(r["correct"] for r in v) / len(v) for k, v in by.items() if len(k) == 3}

    L = ["# Results summary", ""]
    L.append("Generated by `src/analyze.py` from `results/scores.csv`. Headline metric: share of responses whose "
             "final answer is exactly right (all positions / all options). 95% Wilson intervals are over "
             "responses (tasks x samples), so they understate task-level uncertainty.")
    L += ["", "## Overall", "",
          "| Model | Condition | Accuracy | 95% CI | Partial credit | Format failures | Responses | "
          "Tasks solved in every sample |",
          "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for c in conditions:
            v = by[(m, c)]
            k = sum(r["correct"] for r in v)
            lo, hi = wilson(k, len(v))
            solved_all = sum(1 for t in task_ids if by[(m, c, "task", t)] and
                             all(r["correct"] for r in by[(m, c, "task", t)]))
            L.append(f"| {m} | {CONDITION_LABEL[c]} | **{pct(k / len(v))}** | {pct(lo)}-{pct(hi)} | "
                     f"{pct(sum(r['partial'] for r in v) / len(v))} | {sum(1 - r['parse_ok'] for r in v)} | "
                     f"{len(v)} | {solved_all}/{len(task_ids)} |")

    L += ["", "## Accuracy by category", ""]
    for c in conditions:
        L += [f"### {CONDITION_LABEL[c]}", "", "| Category | " + " | ".join(models) + " |",
              "|---|" + "---|" * len(models)]
        for cat in cats:
            L.append(f"| {cat} | " + " | ".join(pct(cat_acc[(m, c, cat)]) if (m, c, cat) in cat_acc else "-"
                                                 for m in models) + " |")
        L.append("")

    L += ["## Accuracy by task", "",
          "Cells: correct samples / samples, closed book then open book.", "",
          "| Task | Category | Difficulty | " + " | ".join(models) + " |",
          "|---|---|---|" + "---|" * len(models)]
    for t in task_ids:
        cells = []
        for m in models:
            parts = []
            for c in conditions:
                v = by[(m, c, "task", t)]
                parts.append(f"{sum(r['correct'] for r in v)}/{len(v)}" if v else "-")
            cells.append(" · ".join(parts))
        L.append(f"| {t} | {tasks[t]['category']} | {tasks[t]['difficulty']} | " + " | ".join(cells) + " |")

    L += ["", "## Why answers were wrong", "",
          "Each wrong answer is compared with the answers produced by deliberately wrong procedures in "
          "`src/engine.py`. A match means the model's answer is exactly what that misconception predicts.", "",
          "| Model | Condition | Wrong answers | Format failure | " +
          " | ".join(["legacy_overall_first", "no_reapply", "restart_h2h", "naive_conduct", "ignore_officials",
                      "gf_before_gd", "conduct_inverted"]) + " | Unexplained |",
          "|---|---|---|---|" + "---|" * 8]
    mis_names = ["legacy_overall_first", "no_reapply", "restart_h2h", "naive_conduct", "ignore_officials",
                 "gf_before_gd", "conduct_inverted"]
    for m in models:
        for c in conditions:
            wrong = [r for r in by[(m, c)] if not r["correct"]]
            cnt = Counter()
            unexplained = 0
            for r in wrong:
                tags = [x for x in r["misconceptions"].split("|") if x]
                cnt.update(tags)
                if r["parse_ok"] and not tags:
                    unexplained += 1
            fmt_fail = sum(1 - r["parse_ok"] for r in wrong)
            L.append(f"| {m} | {CONDITION_LABEL[c]} | {len(wrong)} | {fmt_fail} | " +
                     " | ".join(str(cnt[x]) for x in mis_names) + f" | {unexplained} |")

    figs = [chart_overall(models, conditions, acc, n)] + [chart_heatmap(models, cats, cat_acc, c) for c in conditions]
    L += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs]
    SUMMARY.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY} and {len(figs)} figures")


if __name__ == "__main__":
    main()
