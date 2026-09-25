# Human annotation (P2.3 to P2.6)

Everything runs on your laptop. You need the ChartQA images locally and one
Python package.

## Setup, once

```bash
pip install -r requirements-annotate.txt
scripts/download_chartqa.sh ~/ChartQA          # ~1.1 GiB, images + JSON
```

## Labelling

```bash
git pull
CHARTQA_ROOT=~/ChartQA streamlit run src/label/app.py
```

Type your name in the sidebar (it must match your file in `annotations/tasks/`).
For each item: look at the chart, read the question, the gold answer and the
model's answer, optionally type one line on why, then click Structural,
Fabrication or Ambiguous. The rubric is in the expander at the top and in
`docs/taxonomy.md`.

- Every click is saved to `annotations/<you>.jsonl` immediately. Close the tab
  whenever; reopening resumes where you stopped.
- Misclicked? The "Change to" buttons under the item fix your last answer.
- Label in two or more sittings, not one. Agreement degrades with fatigue and
  the degradation is invisible in the output.
- Do not open anyone else's file, or `annotations/tasks/pool_key.jsonl` (which
  says which model produced each answer), until you have finished yours.

When done: commit and push `annotations/<you>.jsonl`. Each person writes only
their own file, so there are no merge conflicts.

## Checking agreement

Switch the sidebar to **Agreement**. It reads every rater file and reports
Fleiss's kappa against the 0.6 gate, pairwise kappa (one person low against
everyone else means their reading of the rubric has drifted), and every
disagreement with each person's reasoning, for adjudication.

## Setting up the pool (whoever runs it, once inference exists)

```bash
python -m src.label.pool build --split test \
    --source qwen=results/qwen_chartqa_test.jsonl \
    --source llava=results/llava_chartqa_test.jsonl
python -m src.label.pool assign --pool annotations/tasks/pool.jsonl \
    --annotators naman yash shrish sanjith --overlap 200
```

With 1000 wrong answers this gives everyone 300 items: 200 of their own and
100 shared with one teammate. The shared items cycle through all six pairs, so
every person is checked against every other.

## LLM labels (for Yash)

Label the items in `annotations/tasks/pool.jsonl` and write
`annotations/claude.jsonl`, one line per item:

```json
{"item_id": "3fa2c91b0e4d", "label": "structural", "rationale": "..."}
```

Use the `item_id` values from the pool unchanged; that is the only join key.
Labels must be `structural`, `fabrication` or `ambiguous`. The file then shows
up in the Agreement page's pairwise table as human-versus-Claude agreement, and
is excluded from the headline human kappa by default.
