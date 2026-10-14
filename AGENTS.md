# Agent instructions: type-aware VLM failure prediction

ANLP course project, team HiddenStates: Yash More, Naman Singhal, Shrish Kadam,
Sanjith Ganapathi. The mid submission is done (`paper/mid/`). The **final
submission is due 31 Oct 2026**; the team target is 30 Oct.

**Your task list:** before starting work, read `docs/final-plan.md` and find the
section for the person you are working for (ask them if unsure). It holds their
tasks in order, the files involved and the completion criterion for each.
`todos.md` and `TASKS.md` are history; where they disagree with this file or the
plan, this file and the plan win.

## Decisions (binding, taken 14 Oct 2026 by Naman for the team)

Apply these as settled. Raise a disagreement with the team, and keep following
the decision until the team changes it here.

1. **Title.** Drop "and Diagram". The final title is "Type-Aware Failure
   Prediction for Chart Reasoning in Vision-Language Models". Node-link
   diagrams (`src/synth/node_links.py`) are out of scope; the report mentions
   them once, as future work.
2. **`picked_present` is its own outcome, never fabrication.** Naming the shown
   bar in "Which is larger, <missing> or <shown>?" is grouped with "0" answers
   as a wrong answer that invents no value. It stays out of every fabrication
   class and is reported as its own count. This matches the counts already in
   the mid report.
3. **E4 runs on synthetic charts only**, where bar geometry is known: every
   value-question error from pilots 1 and 2, for both models (Qwen 631, LLaVA
   1,015). No ChartQA E4. The test is exactly the one pre-registered in
   `results/e4-preregistration.md` (difference in differences for `I_crop`,
   `I_verify` secondary); the pre-registration is frozen.
4. **Pilot 3** separates fabrication from "the category is missing": bar charts
   whose x-axis is consecutive years, each asked once about a shown year and
   once, in identical wording, about a year outside the shown range. 300 charts,
   both models, the same outcome scoring as pilots 1 and 2 (rejected, "0",
   fabricated).
5. **`not_an_error` items are reported both ways**, all errors and clean, as in
   the mid report. Type probes train on the clean set.
6. **Human agreement check:** 200 errors (100 per model, stratified by the LLM
   label), labelled blind by Shrish and Sanjith in `src/label/app.py`. Cohen's κ
   over the five labels, per model and overall, gate 0.6. The LLM labels remain
   the labels every experiment uses; the human check validates them. A κ below
   0.6 is reported as a limitation, with no relabelling round.
7. **E5** runs on the synthetic value questions with E4 outcomes. Thresholds are
   chosen on the validation split and scored once on test. Baselines:
   binary-only abstention and the label-aware oracle. Metric: risk-coverage
   curves and AURC.

## Rules for everyone

- **Seed 42** for everything in this project, new work included, so runs stay
  comparable.
- **Report and slide prose is written by the team members themselves.** The
  course bans AI-generated reports. Agents produce code, numbers, tables,
  figures and LaTeX structure, and check facts; each person writes the prose of
  their own sections.
- **Every number in the report traces to a file in `results/`.** Commit the
  result file alongside the code that made it, with the command in its header.
- **Cluster details stay out of the repo.** Use `$HOME`, `$PROJ` or a config
  variable in place of any `/home2/<user>` path, node name or cluster hostname.
  The repo is public and is submitted.
- **Ada GPU jobs:** `--constraint=2080ti`,
  `--exclude=gnode066,gnode068,gnode076,gnode077`, and an explicit `--time`
  (the default is 1 hour). gnode066, gnode076 and gnode077 have had broken CUDA
  drivers; `torch` silently falls back to CPU there, so CPU fallback is a hard
  error in every job. Run Python only under `sbatch`/`srun`, never on the login
  node.
- **Git:** short one-line commit messages, batched. Push only when your person
  asks. Pull before starting, since four people share `main`.
- No em dashes in any file.
