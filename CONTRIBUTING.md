# Contributing

Contributions are welcome, especially new tasks, other competitions' rule sets and results for more models.

## Adding a task

1. Add a scenario to [src/build_tasks.py](src/build_tasks.py) using `group_task`, `thirds_task` or
   `scenario_task`. Give it a hand-written `expected` answer and a step-by-step `derivation` that cites the
   Article 13 criteria used.
2. Run `python src/build_tasks.py`. The build fails if your hand-derived answer differs from the reference
   engine. If they differ, work out which one is wrong before changing either.
3. Run `python src/validate_grader.py`. All checks must pass.
4. A good task tests one clause, is fully determined by the information in the prompt, and gives a different
   answer under at least one misconception in `src/engine.py`. If it doesn't, it belongs in `baseline`.

## Adding a model

Add an entry to [config/models.json](config/models.json) (`anthropic` or any `openai_compat` endpoint), run
`python src/run_models.py --models <label>`, then `grade.py` and `analyze.py`. Commit the raw outputs in
`results/raw/<label>/` so others can re-grade them.

## Reporting a rules disagreement

If you think an expected answer misreads the regulation, open an issue quoting the clause and your derivation.
