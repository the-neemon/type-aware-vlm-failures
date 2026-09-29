# To-dos after the mid-submission

Written 29 Sep 2026. Ordered by importance within each section. Cost: **CPU** =
minutes on a compute node from existing caches; **GPU** = a new Qwen or LLaVA
run; **labelling** = people's time. Results so far: `paper/mid/main.pdf`,
`results/synth_type_probes.md`, `results/probes/`.

## Before the 30 Sep submission

- [ ] **Finish the report** (`paper/mid/main.tex`). Two table fixes pending:
      Table 3 and Table 4 are a few points too wide (add
      `\setlength{\tabcolsep}{4pt}` after `\centering\small`), and Table 4's
      `\multirow{4}{*}{Missing, value}` should span 3 rows. Proofread; each
      member submits individually.
- [ ] **Commit and merge.** `paper/mid/` and two result files are uncommitted:
      `results/probes/binary_without_not_an_error_qwen2_5_vl_7b_test.json` and
      `results/probes/structural_chartqa_qwen2_5_vl_7b_test.json`. Merge
      `llava-next` into `main`.
- [ ] **Repo access for graders.** Public, or TA mentors granted access, before
      30 Sep; HuggingFace or WandB links in the README (TASKS.md checklist).

## Core plan still owed (October)

- [ ] **E4, cross-intervention matrix** (GPU; code exists in `src/intervene/`).
      Re-query every synthetic error with a question-conditioned crop, a
      "verify your region" prompt, and abstention; record recovery by failure
      type, for both models. Expected if the taxonomy matters: misreads recover
      with cropping (especially LLaVA's gridline rounding), missing-bar
      fabrications do not. The whole "type-aware" argument rests on this.
- [ ] **E5, controller** (CPU, after E4). Missing-bar probe -> abstain; binary
      and misread probes -> re-look. Risk-coverage and AURC against binary-only
      abstention and the label-aware oracle.
- [ ] **Human agreement check** (labelling). Two people label the same 200 Qwen
      errors blind; kappa > 0.6. The only validation the ChartQA labels have.
- [ ] **Finish E2** (CPU, small code changes):
  - [ ] residualised probes (Sahoo et al.): probe accuracy after regressing
        surface features out of the activations;
  - [ ] a surface baseline for the ChartQA structural probe;
  - [ ] add "value position relative to gridlines" to the synthetic metadata
        baseline, declared in advance this time (it predicts LLaVA's misreads
        at 0.852, above its probe).

## Cheap experiments that would strengthen the paper (CPU, existing caches)

- [x] **Synthetic -> ChartQA transfer.** Train the misread and missing-bar
      probes on synthetic charts, apply them to real ChartQA items. Does the
      synthetic misread probe flag Naman's 71 structural errors? The strongest
      test that the synthetic findings mean anything for real charts.
      Code done (`python -m src.probes.run_transfer transfer`); results pending
      a run on the cluster once maintenance ends.
- [ ] **Computation as a third class.** Largest class of real errors (93 of
      Qwen's 317). Can a probe tell computation errors from misreads?
- [ ] **Use val_human** (960 Qwen questions, 193 errors, no chart shared with
      test) as extra training data or a second held-out test.

## New data that would close gaps (GPU)

- [ ] **Separate "fabrication" from "the category is missing"**, currently the
      same by construction: out-of-range questions (e.g. a year the chart does
      not show, as the proposal planned), or questions needing information the
      chart lacks (units, data source).
- [ ] **More Qwen misreads** (only 46). 0-200 axes with sparse ticks gave a 40%
      misread rate; ~1,000 such charts would give a few hundred.
- [ ] **Node-link diagrams**, the "diagram" half of the title. Generator exists
      (`src/synth/node_links.py`), never run. Misreads = wrong path tracing;
      fabrications = answers about nodes that do not exist.
- [ ] **LLaVA's 1,178 ChartQA errors labelled** (labelling, expensive), then a
      LLaVA structural probe on real charts. Only if the paper needs it.

## Optional, higher risk

- [ ] **Causal test.** Steer activations along the missing-bar direction; does
      the model start refusing? Yuan et al. found such signals can be
      diagnostic but not causal.
- [ ] **The "0" diagnostic.** Rerun the absent questions without "Answer the
      question using a single word or phrase." to confirm "0" means "no bar".
      Diagnostic only: those activations are not comparable with the rest.
- [ ] **Robustness.** Main probes over several random splits; check calibration
      before the controller relies on probe scores.

## Housekeeping

- [ ] Decide whether naming the shown bar in "Which is larger, <missing> or
      <shown>?" (`picked_present`) counts as fabrication.
- [ ] Fix the 3 failing item_id tests (`src/extract/test_cache_io.py` x2,
      `src/label/test_pool.py::test_item_id_is_the_inf_md_scheme`).
- [ ] Delete leftover shard folders on Ada once their runs are confirmed:
      `~/activations/*/shards_*` in Shrish's home (Qwen ~4.5 GB, plus LLaVA's
      `shards_synth_pilot` and `shards_synth_pilot2`). The final `.npz` files
      stay.

**If only two:** E4 (the experiment the thesis rests on) and synthetic ->
ChartQA transfer (minutes of CPU, and it says whether the synthetic results
generalise).
