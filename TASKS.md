# TASKS: Execution Plan

Companion to [SPEC.md](SPEC.md). SPEC says what we are building and what counts
as done. This file says who does what, in what order, and what will bite us.

**Today: Wednesday 2 September 2026.**
**Mid submission: Wednesday 30 September 2026 (4 weeks out).**
**Final submission: Saturday 31 October 2026 (8.5 weeks out).**

---

## 0. Critical Path

Four things gate everything downstream. If one slips, the phase behind it slips.

1. **The pooling decision** (P0.4) gates all activation caching. Get it wrong and
   we need terabytes we do not have. Decide before caching a single item.
2. **Error yield** (P1) gates E4 statistical power. We need roughly 400 labelled
   errors per failure type to get usable confidence intervals in a 2 x 4 matrix.
   ChartQA test alone will not produce that. Plan train-split runs now.
3. **The kappa gate** (P2.4) blocks every probe experiment on the naturalistic
   arm. If it fails we lose a week re-defining categories and re-annotating.
4. **The 30 September go/no-go** decides whether the back half is E5 or a
   negative-result write-up. Do not let this date arrive undecided.

The synthetic arm (P3) is deliberately off the critical path and runs in
parallel. It is the insurance policy for items 2 and 3.

## 1. Workstream Ownership

Four people, four workstreams. Rotate the write-up load; do not let one person
own the paper alone.

| Workstream | Scope | Owner |
| --- | --- | --- |
| **A. Infra and inference** | Ada, SLURM, envs, model runs, activation caching | TBD |
| **B. Labelling** | Judge pipeline, annotation, kappa, taxonomy definitions | TBD |
| **C. Synthetic** | Generator, matched pairs, ground-truth labels | TBD |
| **D. Analysis** | Probes, surface control, matrix, controller, plots | TBD |

Assign these at the next team meeting and record them here. Every member needs
enough context on workstream B to annotate, since the kappa check needs two
independent annotators.

---

## 2. Phase Breakdown

### P0. Infrastructure (2 Sep to 8 Sep, 1 week)

- [ ] **P0.1** Confirm Ada access for all four members. Verify each can submit a
      job and write to `/scratch`.
- [ ] **P0.2** Pin the environment. Qwen2.5-VL needs a recent `transformers` plus
      `qwen-vl-utils`; LLaVA-NeXT has its own processor path. Lock versions in
      `requirements.txt` and record the exact commit. Version drift between team
      members will silently change activations.
- [ ] **P0.3** Set `HF_HOME` to `/scratch`, not home. Home quota will not hold two
      7B checkpoints (roughly 16 GB each). Download both checkpoints once and
      share the path.
- [ ] **P0.4** **Decide the caching schema and compute the storage budget before
      writing any cache.** This is the single most consequential infra decision
      in the project. See Section 4.1 for the arithmetic.
- [ ] **P0.5** SLURM job template. Request explicit walltime; the cluster default
      is 1 hour and a full ChartQA pass will not finish in it. Pin GPU type in the
      constraint, since nodes are mixed and a job that lands on the wrong card
      will OOM or run slow. Avoid `gnode077` (known dead GPU).
- [ ] **P0.6** Repo skeleton per SPEC Section 10: `configs/`, `src/`, `scripts/`,
      `results/`, `paper/`. Add `.gitignore` for caches, checkpoints and `*.npz`.
- [ ] **P0.7** README stub. Guidelines require HuggingFace and WandB links live
      here. Add the section now even if empty.
- [ ] **P0.8** Grant TA mentor access to the repo, or make it public. Required by
      30 September; do it now so it cannot be forgotten.

### P1. Inference and error collection (5 Sep to 14 Sep, 1.5 weeks)

- [ ] **P1.1** Implement ChartQA's **relaxed accuracy** metric (5 percent
      tolerance on numeric answers, exact match on strings). Do not use exact
      match. See Section 4.2.
- [ ] **P1.2** Fix the decoding config: greedy, temperature 0, fixed seed, capped
      max tokens. Record it in `configs/`. Every downstream label refers to a
      specific answer string, so the answer must be reproducible.
- [ ] **P1.3** Run Qwen2.5-VL-7B over ChartQA `human_test` and `augmented_test`.
      Record accuracy and sanity-check it against the published number for the
      model. A large gap means the prompt or the metric is wrong, not the model.
- [ ] **P1.4** Same for LLaVA-NeXT.
- [ ] **P1.5** Count errors per model. If the combined yield is under roughly 800,
      extend to the ChartQA train split until the yield clears 1500. Budget a
      day of GPU time for this.
- [ ] **P1.6** Store `(figure_id, question, gold, prediction, correct)` rows as
      the single source of truth that labelling and caching both key off.

### P2. Labelling (8 Sep to 21 Sep, 2 weeks, overlaps P1)

- [ ] **P2.1** Write the two-class definitions with worked edge cases. Include an
      explicit **"neither / ambiguous"** escape hatch so annotators and the judge
      are never forced to pick. Ambiguous items are dropped, not coerced.
- [ ] **P2.2** Build the VLM judge pipeline. The judge sees figure, question, gold
      answer and model answer. Estimate API cost before launching; images make
      this the largest cash cost in the project.
- [ ] **P2.3** Pilot the judge on 50 items. Read every one by hand. Refine the
      definitions. Expect at least two rounds.
- [ ] **P2.4** **Kappa gate.** Two team members independently annotate roughly
      200 items. Compute Cohen's kappa. Threshold is 0.6.
      - Pass: proceed.
      - Fail: simplify definitions, adjudicate, re-annotate. Do not proceed on
        unreliable labels. Budget a full week for a failure.
- [ ] **P2.5** **Also measure judge-versus-human agreement** on the same 200
      items. Inter-human kappa validates that the taxonomy is clear; it says
      nothing about whether the judge applies it correctly. Both numbers go in
      the paper. See Section 5.1.
- [ ] **P2.6** Run the judge over the full error set. Record per-item judge
      rationale for the appendix.

### P3. Synthetic arm (5 Sep to 21 Sep, parallel, 2 weeks)

- [ ] **P3.1** Bar chart generator (matplotlib) with controllable bar-height
      separation, tick density and axis range.
- [ ] **P3.2** Node-link diagram generator (networkx plus a layout) with
      controllable path length and edge crossing count.
- [ ] **P3.3** Question templates that induce each failure type by construction:
      near-identical bar heights for structural, out-of-range or absent-category
      queries for fabrication.
- [ ] **P3.4** **Matched pairs.** Items that hold visual precision fixed while
      varying relational depth, and the converse. This is what lets us separate
      the two factors by design rather than post hoc, and it is the synthetic
      arm's main scientific contribution. Do not skip it under time pressure.
- [ ] **P3.5** Figure-level split assignment, written into the manifest at
      generation time so it cannot drift.
- [ ] **P3.6** Generate at scale with class balance set deliberately, targeting at
      least 400 items per failure type.

### P4. Activation caching (14 Sep to 24 Sep, 1.5 weeks)

- [ ] **P4.1** Implement forward hooks at the chosen layers and token positions.
- [ ] **P4.2** **Verify the hooks fire where you think they do.** Assert on tensor
      shapes and layer indices. A silently misindexed hook produces a clean null
      result that looks like a scientific finding.
- [ ] **P4.3** Cache activations for every item in the error set and a matched
      sample of correct items (the binary probe needs both classes).
- [ ] **P4.4** Couple caches to answers. Either cache during the same forward pass
      that produced the labelled answer, or re-run and assert the regenerated
      answer matches the stored one byte for byte.
- [ ] **P4.5** **Pipeline sanity probe.** Train a probe to predict something
      trivially decodable from the activations, such as figure type or question
      template. If that does not hit high AUROC, the caching is broken and every
      downstream null is meaningless. Run this before trusting any negative
      result.

### P5. Probes and controls (20 Sep to 30 Sep, 1.5 weeks): E1, E2, E3

- [ ] **P5.1** Binary probe (correct vs incorrect), per-layer AUROC, both models.
      Check against HALP's reported range as the sanity condition.
- [ ] **P5.2** Structural and fabrication one-vs-rest probes, per-layer AUROC.
- [ ] **P5.3** Layer selection on **validation only**. Write the selection code so
      test data is not even loaded during selection.
- [ ] **P5.4** **E2 surface baseline.** Classifier over surface features alone:
      answer present in figure text, answer type, answer length, question
      template, figure type. Report the margin the activation probe exceeds it by.
- [ ] **P5.5** **E2 residualisation.** Probe accuracy after residualising the
      surface features out, following Sahoo et al. This is the number that
      decides whether we have a mechanism or a confound.
- [ ] **P5.6** **E3 cross-transfer.** Train each type probe on its own class, test
      on the other. Strong transfer means one signal wearing two labels, and that
      is a finding, not a failure.
- [ ] **P5.7** Bootstrap confidence intervals on every reported number. Cluster
      the bootstrap by figure, not by item, since figures contribute multiple
      questions.

### M1. Mid submission (24 Sep to 30 Sep): HARD DEADLINE

- [ ] **M1.1** Write the 7 to 8 page document, ACL style, excluding references,
      optional single appendix page. **Written by humans. The guidelines
      explicitly forbid AI-generated reports.** AI assistance on code is fine.
- [ ] **M1.2** Comprehensive literature review section. This is graded and it is
      the section most often under-weighted.
- [ ] **M1.3** Formal refined problem statement plus concrete remaining timeline.
- [ ] **M1.4** Confirm repo is public or TA mentors have access.
- [ ] **M1.5** README carries any HuggingFace and WandB links.
- [ ] **M1.6** Package `HiddenStates-Mid.zip`. **Every team member submits
      individually.**
- [ ] **M1.7** Prepare for viva. Slides optional but recommended.

### GO / NO-GO: 30 September

Decide and record in the repo:

- Probes separate **and** survive E2 → back half is E5, the full controller.
- Matrix work proceeds regardless; E4 is independent of E1 to E3.
- Probes fail E2 → back half documents the negative result and leans on E4.

### P6. Cross-intervention matrix (1 Oct to 15 Oct, 2 weeks): E4

- [ ] **P6.1** Nail down the `I_0` definition. See Section 5.2; this is currently
      under-specified.
- [ ] **P6.2** Implement `I_crop`: re-query at higher resolution on a
      **question-conditioned** crop. The conditioning mechanism needs a decision
      (Section 5.3). The crop must never be conditioned on the model's own
      attention.
- [ ] **P6.3** Implement `I_verify`: re-query with a prompt demanding the model
      name the supporting figure region.
- [ ] **P6.4** Implement `I_abstain`, scored as risk avoided, not accuracy gained.
- [ ] **P6.5** Run all four interventions across both failure types, both models,
      both data arms.
- [ ] **P6.6** Bootstrap confidence intervals per cell, clustered by figure.
- [ ] **P6.7** Test for diagonal structure explicitly. State the test in advance
      rather than eyeballing the matrix.

### P7. Controller and utility (15 Oct to 25 Oct, 1.5 weeks): E5

- [ ] **P7.1** Oracle policy (sees true failure type). This is the ceiling.
- [ ] **P7.2** Learned controller (activations only).
- [ ] **P7.3** Binary selective-prediction baseline.
- [ ] **P7.4** Risk-coverage curves and AURC for all four settings at **matched
      coverage**. Comparing at unmatched coverage is meaningless.
- [ ] **P7.5** Ablation: does re-querying contribute anything beyond abstention?

### M2. Final submission (20 Oct to 31 Oct): HARD DEADLINE

- [ ] **M2.1** Write-up, at most 8 pages excluding references, up to 2 appendix
      pages, structured as a research paper. Human-written.
- [ ] **M2.2** Every claim backed by prior or empirical results with analysis.
- [ ] **M2.3** **Limitations and future work in a separate section immediately
      before the references, containing nothing else.** Any project result placed
      there gets graded as future work. This is a specific, easily-tripped rule.
- [ ] **M2.4** Presentation PDF for the viva.
- [ ] **M2.5** Code ZIP inside the submission.
- [ ] **M2.6** README final pass with all HuggingFace and WandB links.
- [ ] **M2.7** Package `HiddenStates-Final.zip`. **Every member submits.**

---

## 3. Milestone Calendar

| Week | Dates | Focus | Milestone |
| --- | --- | --- | --- |
| 1 | 2 Sep to 8 Sep | P0 infra, P1 starts, P3 starts | Storage schema decided |
| 2 | 9 Sep to 14 Sep | P1 runs, P2 judge build, P3 generators | Error yield known |
| 3 | 15 Sep to 21 Sep | P2 kappa gate, P4 caching, P3 at scale | **Kappa gate passed** |
| 4 | 22 Sep to 30 Sep | P5 probes and controls, M1 write-up | **Mid submission, go/no-go** |
| 5 to 6 | 1 Oct to 15 Oct | P6 cross-intervention matrix | E4 matrix complete |
| 7 | 16 Oct to 25 Oct | P7 controller, risk-coverage | E5 complete |
| 8 | 26 Oct to 31 Oct | M2 write-up, slides, packaging | **Final submission** |

Slack is thin. Week 3 is the tightest, since the kappa gate, caching and
synthetic scale-up all land together. If something has to give, it is the
ChartGemma stretch goal, and it should be dropped without discussion.

---

## 4. Things To Be Careful About: Infrastructure

### 4.1 Activation storage will explode if you cache raw vision tokens

Qwen2.5-VL-7B has hidden dimension 3584. A chart at reasonable resolution yields
on the order of 1500 vision tokens after patch merging.

- **Pooled**, at 8 layers and 2 positions, fp16: roughly 115 KB per item. Twenty
  thousand items is about 2 GB. Comfortable.
- **Unpooled vision tokens**, 8 layers, fp16: roughly 90 MB per item. Twenty
  thousand items is about 1.8 TB. Not viable.

Decide the pooling strategy (mean over vision tokens, last query token, or a
small fixed set of pooled statistics) in P0.4, write it into `configs/`, and do
not change it mid-project. Store fp16, not fp32. If a later experiment needs
unpooled tokens, re-cache a small subsample rather than everything.

### 4.2 ChartQA scoring is relaxed accuracy, not exact match

The benchmark allows 5 percent tolerance on numeric answers. Scoring with exact
match will mark correct answers as errors, and those mislabelled items will flow
straight into the "structural misreading" class and quietly poison every probe.
Implement the official metric and unit-test it on the tolerance boundary.

### 4.3 Error scarcity is a real risk

Strong VLMs sit around 75 to 85 percent on ChartQA. The test splits alone yield
only a few hundred errors per model, and fabrications will be the minority of
those. E4 needs roughly 400 labelled errors per failure type for tight cells.
Plan train-split inference from the start rather than discovering the shortfall
in October.

### 4.4 Ada specifics

Request explicit walltime; the default is one hour. Nodes carry mixed GPU types,
so constrain the GPU in the job script or a large model will land on a card that
cannot hold it. `/scratch` persists across jobs, so cache there and not in the
job's temp space. Avoid `gnode077`.

### 4.5 Reproducibility hygiene

Fix seeds, pin library versions, and write the full run config next to every
results file. Two team members running "the same" job on different transformers
versions will produce different activations and neither will notice.

---

## 5. Things To Be Careful About: Methodology

### 5.1 Inter-human kappa does not validate the judge

The proposal sets kappa > 0.6 between two team members. That measures whether
the taxonomy is *definable*. It does not measure whether the LLM judge *applies*
it correctly, and the judge is what labels the other ninety percent of the data.
Report both inter-human kappa and judge-versus-human-consensus agreement. If the
judge agrees with humans much less than humans agree with each other, the label
noise is in the judge and the probes will underperform for reasons that have
nothing to do with the hypothesis.

### 5.2 `I_0` needs a real definition

Under greedy decoding, re-running the same prompt reproduces the same wrong
answer, so a naive `I_0` recovery rate is zero by construction and makes every
other intervention look good for free. Either define `I_0` as "keep the original
answer" and treat the zero as a definitional floor, or add a **naive re-ask**
control (same prompt, resampled) so that the matrix separates "this repair
works" from "asking twice works". Decide before running P6 and state it in the
paper.

### 5.3 `I_crop` conditioning is under-specified

SPEC says the crop is question-conditioned and explicitly not
attention-conditioned, which is the right constraint, but it does not say *how*
the crop is derived. Options, in increasing order of cost:

1. Uniform higher-resolution re-render of the whole figure. Weakest, but a clean
   control that isolates resolution from localisation.
2. OCR the figure, match question tokens against detected text, crop to the
   matched region plus margin.
3. Use gold evidence-region annotations where the source dataset provides them.

Option 1 is the honest floor and should be run regardless, because without it a
gain from option 2 cannot be attributed to localisation rather than pixels.

### 5.4 Splits must be figure-level, and layer selection must be validation-only

ChartQA has multiple questions per figure. Splitting by question leaks the figure
across train and test and will inflate every probe. Assert on this in code rather
than trusting the split script. Separately, selecting the best layer by test
AUROC is the classic version of the same mistake; the selection code should never
load test data.

### 5.5 Bootstrap by figure, not by item

Items sharing a figure are correlated. Item-level bootstrap will produce
confidence intervals that are too narrow, which matters most in E4 where the
whole claim is about whether cells differ.

### 5.6 A broken pipeline looks exactly like a negative result

This project is designed so that null results are publishable, which makes it
unusually vulnerable to shipping a bug as a finding. The P4.5 sanity probe is the
defence: if the activations cannot predict something trivial like figure type,
they cannot predict anything, and the null is about the code. Run it before
writing any sentence containing the phrase "does not separate".

### 5.7 Do not pool results across models

HALP reports that the most informative representation family varies by
architecture, with Qwen2.5-VL-7B best served by visual-only features where other
models rely on late query-token states. Report Qwen and LLaVA separately
throughout. An averaged number across two architectures with different optimal
representations is not meaningful.

### 5.8 Keep the naturalistic arm central

The synthetic arm is easier, cleaner and more fun to work on, and it will try to
eat the project. It is a control and a source of balanced classes. The claim is
about real chart reasoning. If the paper's headline numbers come from
synthetic data, the contribution shrinks accordingly.

### 5.9 Report the gap, never the raw AUROC alone

Every probe number in every table, slide and figure needs its surface baseline
next to it. A raw 0.85 AUROC with no control is not evidence and reviewers, TAs
and the viva panel will all ask the same question.

---

## 6. Course Compliance Checklist

Easy marks, easily lost.

- [ ] ACL Style Guide for both write-ups.
- [ ] **Reports written by humans.** AI is permitted for implementation and code
      only. This is an explicit course rule.
- [ ] Every team member submits every deliverable individually.
- [ ] Filenames exact: `HiddenStates-Mid.zip`, `HiddenStates-Final.zip`.
- [ ] Page limits: mid 7 to 8 plus 1 appendix; final at most 8 plus 2 appendix.
      References excluded from both counts.
- [ ] Repo public or TA mentors granted access, before 30 September.
- [ ] HuggingFace and WandB links in the README.
- [ ] Final report's limitations and future work section sits immediately before
      the references and contains only that.
- [ ] Presentation PDF included in the final ZIP.
- [ ] No extensions exist. Both deadlines are hard.

## 7. Open Decisions

Record the resolution here as each is made.

| # | Decision | Needed by | Status |
| --- | --- | --- | --- |
| 1 | Pooling strategy and cached layer set | 8 Sep | Open |
| 2 | Judge model and cost ceiling | 10 Sep | Open |
| 3 | Whether to run ChartQA train split for error yield | 12 Sep | Open |
| 4 | Workstream ownership | Next meeting | Open |
| 5 | `I_0` definition, with or without a naive re-ask control | 1 Oct | Open |
| 6 | `I_crop` conditioning mechanism | 1 Oct | Open |
| 7 | ChartGemma stretch goal: keep or drop | 30 Sep | Open |
