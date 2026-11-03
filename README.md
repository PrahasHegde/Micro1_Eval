# Tiebreak: an eval of FIFA World Cup 26 group-ranking rules

**Short description.** Ranking a football group when teams finish level on points is a small, fully specified
legal procedure, and the rules changed for the 2026 World Cup. Article 13 of the FIFA World Cup 26 Regulations
now applies **head-to-head results first**, re-applies them to any sub-group that is still tied, then moves to
overall goal difference, goals, a card-based conduct score and finally the world ranking. It also has a
"does not restart" clause. The 2018 and 2022 World Cups used **overall goal difference first** and ended with
drawing of lots. Tournament analysts, broadcasters and federation staff get this right with a pen and paper.
This eval measures whether language models can. It has 19 expert-derived tasks, exact expected answers,
a deterministic grader (no LLM judge), and automatic diagnosis of *which* wrong procedure a model followed.

**Key findings:** _pending the first run; see [Models tested and results](#6-models-tested-and-results)._

---

## 1. Video walkthrough

_Link to be added._ The narration script (problem, gap, demo, findings) is in
[docs/video_script.md](docs/video_script.md), written to be read by an AI voice.

## 2. Domain and expertise

* **Domain:** football competition regulations, specifically group-stage tie-breaking and qualification
  for the FIFA World Cup 26 (48 teams, 12 groups, the 8 best third-placed teams advance).
* **Primary source:** `FWC26_regulations_EN.pdf` (FIFA World Cup 26 Regulations), Article 13, pp. 26-27.
  The verbatim text used in the open-book condition is in [prompts/fwc26_article13.txt](prompts/fwc26_article13.txt).
* **Expertise applied:** each task was designed to exercise one clause of Article 13. Each expected answer
  was derived by hand, step by step (see `derivation` in every task), and then independently re-derived by a
  reference implementation of the regulation. A task is only accepted when both agree.

## 3. What it tests and why it matters

**Who it is for.** Teams building assistants for sports media, betting and fantasy products, federation and
league operations, and anyone who uses an LLM to answer "what does my team need on the last matchday?".
Those questions are asked by millions of people at every tournament, under time pressure, and wrong answers
get published.

**The gap.** Models handle simple tables well. They fail when the procedure has several steps or differs from
the version they memorised:

| Category | Tasks | What has to be done right |
|---|---|---|
| `baseline` | B1-B3 | Controls: no ties, or ties where every rule order agrees. Shows the model can read a table. |
| `h2h_vs_overall` | H1-H3 | Head-to-head result beats a better overall goal difference (changed in 2026). |
| `three_way` | T1-T3 | Mini-table of the tied teams; **step 2 re-application** to a still-tied pair. |
| `no_restart` | N1-N2 | Once in step 2, the remaining teams are *not* compared head-to-head again. |
| `conduct` | C1-C2 | Conduct score: a second yellow is -3 (not -5), officials' cards count, one deduction per person per match. |
| `world_ranking` | R1 | Step 3: joint position in the latest ranking, so use the preceding edition. Lots no longer exist. |
| `four_way` | F1 | All four teams level, so the head-to-head table is the whole group; criterion c) settles it before any pairwise comparison. |
| `third_place` | P1-P2 | Ranking the 12 third-placed teams (no head-to-head); P2 first requires a three-way tie-break inside Group L. |
| `scenario` | S1-S2 | Final-matchday permutations: which scorelines send a team through. |

Every task lists its teams, results, the full disciplinary record and the FIFA ranking, so it is fully
determined. Teams are fictional, so models cannot fall back on memorised real tables.

**Two conditions isolate the cause of failure:**

* **Closed book:** "apply the FIFA World Cup 26 regulations". This tests whether the model knows the current rules.
* **Open book:** the same prompt plus the verbatim Article 13 text. This tests whether it can *execute* the
  procedure when the rules are in front of it.

A model that fails closed book but passes open book has outdated knowledge. A model that fails both has a
procedural-reasoning gap.

## 4. Setup and run instructions

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then paste your API key(s) into .env
```

**Open-source models (default, no API key needed).** Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen3:4b
ollama pull qwen2.5:7b
ollama pull llama3:latest
```

The runner talks to Ollama at `http://localhost:11434` (override with `OLLAMA_HOST`).

**Hosted models (optional).** Choose models in [config/models.json](config/models.json) (`enabled: true/false`).
Anthropic models use the Anthropic SDK. Any OpenAI-compatible endpoint (OpenAI, Gemini, OpenRouter, vLLM)
uses `openai_compat`. Put the keys in `.env`.

```bash
python src/build_tasks.py         # (optional) rebuild tasks/tasks.jsonl; fails if any hand-derived answer disagrees with the engine
python src/validate_grader.py     # engine + grader self-checks (no API calls)
python src/run_models.py --samples 1 --workers 1   # query models -> results/raw/ (resumable; --dry-run to preview)
python src/grade.py               # results/raw -> results/scores.csv
python src/analyze.py             # -> results/summary.md + results/figures/
python src/validate_grader.py     # re-run: also checks determinism on real outputs and writes results/grader_audit.csv
```

Useful flags: `--models claude-opus-5-5 --tasks H1,T2 --samples 1 --conditions open_book`.

## 5. Scoring logic

**Fully automated and deterministic. No model is used as a judge.**

1. Each prompt asks the model to finish with `FINAL ANSWER: {"ranking": [...]}` (or `{"options": [...]}`).
2. [src/grade.py](src/grade.py) reads the object after the **last** `FINAL ANSWER` marker. It accepts JSON or
   single-quoted literals, any capitalisation, markdown bold and code fences. Team names are matched
   case-insensitively.
3. Scoring per task type:

| Type | `correct` (headline, 0/1) | `partial` (context only) |
|---|---|---|
| group ranking | exact 1st-4th order | share of positions right |
| third-placed ranking | exact order of the 8 qualifiers | share of the 8 qualifiers identified |
| scenario | exact set of option letters | Jaccard overlap |

4. A response with no parseable final answer scores 0 and is counted separately as a *format failure*, so
   formatting problems are never confused with reasoning errors.
5. **Failure diagnosis:** [src/engine.py](src/engine.py) also implements deliberately wrong procedures:
   `legacy_overall_first` (2018/2022 order), `no_reapply`, `restart_h2h`, `naive_conduct`, `ignore_officials`,
   `gf_before_gd`, `conduct_inverted`. Each task stores what each misconception would answer, and wrong model
   answers are tagged when they match one exactly.
6. **Aggregation:** accuracy = mean of `correct` over tasks × samples (3 samples per task by default), with
   Wilson 95% intervals. Also reported: tasks solved in every sample (consistency), per-category and
   per-task accuracy.

**How the answer key and the grader were checked** ([src/validate_grader.py](src/validate_grader.py), all passing):

| Check | Result |
|---|---|
| Engine reproduces real World Cup tables under pre-2026 rules (2018 Group H fair-play case, 2022 Groups E and H) | 3/3 |
| Engine agrees with an independently structured second implementation of Article 13 on 5,000 random groups (2,794 with ties, 741 reaching step 2 or later) | 5,000/5,000 |
| Engine output is invariant to the order teams, matches and cards are listed | 5,000/5,000 |
| Every hand-derived expected answer reproduced by the engine | 19/19 |
| Grader battery: correctly formatted right answers must pass (156 cases) | 0 false fails |
| Grader battery: misconception answers, swaps, omissions, duplicates, revised-to-wrong, malformed (225 cases) | 0 false passes |
| Grading the same raw outputs twice | byte-identical (checked after each run) |
| Manual audit of a seeded random sample of 40 graded real responses | `results/grader_audit.csv` |

## 6. Models tested and results

_Not run yet._ After `run_models.py` → `grade.py` → `analyze.py`, this section is filled from
[results/summary.md](results/summary.md). The run date, model IDs (requested and as returned by the API),
parameters, git commit and task-file hash are recorded in `results/run_manifest.json` and in every raw output.

Default configuration: Claude Opus 5.5 (`claude-opus-5-5`), Claude Sonnet 5.5 (`claude-sonnet-5-5`) and
Claude Haiku 5.5 (`claude-haiku-5-5`), all with adaptive thinking at effort `high`, `max_tokens` 32,000, and the
same system prompt ([prompts/system_prompt.txt](prompts/system_prompt.txt)). Other providers can be enabled
in `config/models.json`. These models do not accept a temperature setting, so each task is sampled 3 times
and both the mean and the all-samples consistency are reported.

## 7. Failure analysis

_Pending the run._ The analysis will be based on the "Why answers were wrong" table in `results/summary.md`,
the closed-book vs open-book comparison, and reading the raw reasoning in `results/raw/` for the hardest tasks.

## 8. Prior work, benchmarks and data

* **Rules source:** FIFA World Cup 26 Regulations (Article 13). Pre-2026 behaviour was checked against the
  published final tables of the 2018 and 2022 World Cups.
* **Related benchmarks:** sports QA datasets (e.g. SportQA, 2024) test sports *knowledge* through multiple
  choice; table-reasoning benchmarks (WikiTableQuestions, TabFact) test lookups and simple aggregation on
  tables. Neither tests executing a multi-step, versioned rule procedure with exact, verifiable output.
  Legal and regulatory reasoning sets (LegalBench) are the closest in spirit; this eval adds a domain where
  ground truth is computable, so grading needs no human or LLM judgement.
* **Data:** all teams and results are synthetic and authored for this eval, so there is no training-data
  contamination with real tournament tables. No external dataset is used.

## License and contributing

Code, tasks, prompts and results are open source under the [MIT License](LICENSE). The FIFA regulations PDF
and the quoted Article 13 text remain FIFA's and are included for reference only. New tasks, rule sets and
model results are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository layout

```
tasks/tasks.jsonl          19 tasks: prompts (both conditions), expected answer, hand derivation, misconception answers, raw data
src/engine.py              reference implementation of Article 13 + misconception variants
src/build_tasks.py         scenario definitions -> tasks.jsonl (asserts hand derivation == engine)
src/run_models.py          query models, store raw outputs
src/grade.py               deterministic grader -> results/scores.csv
src/analyze.py             tables + figures -> results/summary.md
src/validate_grader.py     engine and grader validation
prompts/                   system prompt and Article 13 text
config/models.json         models and parameters
results/                   raw outputs, scores, summary, figures, run manifest, grader audit
```
