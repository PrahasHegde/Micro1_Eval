# Video walkthrough: narration script (about 4 minutes)

Read by an AI voice. Bracketed lines are what to show on screen.

---

**[Title card: "Tiebreak: can AI rank a World Cup group?"]**

At every World Cup, millions of people ask the same question: if my team finishes level on points, do we go
through? The answer is written down in the tournament regulations, and it is exact. There is exactly one
right ranking.

**[Show page 26 of the FWC26 regulations, Article 13]**

For the 2026 World Cup, FIFA changed the rule. In 2018 and 2022, tied teams were separated by overall goal
difference first. From 2026, head-to-head results come first. If three teams are tied and only one
separates, you re-apply head-to-head to the two that remain. Only then do you move on to goal difference,
goals scored, a card-based conduct score and finally the world ranking. Once you reach that second stage,
the procedure never restarts.

**[Show the task H1 prompt]**

That is a procedure an analyst can follow perfectly with a pen and paper. So I built nineteen scenarios with
fictional teams, each designed around one clause: head-to-head against goal difference, three-way
mini-tables, the no-restart clause, second yellow cards, the world-ranking fallback, the best third-placed
teams and final-matchday permutations. Three are easy controls where every version of the rules agrees.

**[Show build_tasks.py derivation and the engine assertion]**

Every answer was derived by hand, then checked against a reference implementation of Article 13. That
implementation reproduces real World Cup tables, and on five thousand random groups it agrees with a second,
independently written implementation.

**[Show a raw model response and the FINAL ANSWER line]**

Each model gets every task twice. Closed book, it must know the 2026 rules. Open book, the article is pasted
in, so it only has to follow it. Grading is pure code, with no AI judge. The final answer is either exactly
right or it is not, and the grader was tested on almost four hundred synthetic answers with zero false passes
and zero false fails.

**[Show results/figures/accuracy_by_model.png]**

Here are the results. *(Read the headline numbers from results/summary.md.)*

**[Show the "Why answers were wrong" table]**

The most useful part is *why* models fail. For every wrong answer, the grader checks whether it matches a
known wrong procedure, such as the old goal-difference-first order, skipping the re-application step, or
counting a second yellow as minus five. *(Read the top two misconceptions and the closed vs open-book gap.)*

**[Show README sections 4-5]**

Everything is reproducible: one command builds the tasks, one runs the models, one grades, and every raw
output is saved with the model version and date. Thanks for watching.
