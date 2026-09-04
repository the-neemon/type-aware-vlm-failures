# SPEC: Type-Aware Failure Prediction for Chart and Diagram Reasoning in VLMs

**Course:** Advanced NLP, Monsoon 2026, IIIT Hyderabad
**Team:** Yash More, Naman Singhal, Shrish Kadam, Sanjith Ganapathi
**Repository:** https://github.com/the-neemon/type-aware-vlm-failures
**Status:** Proposal accepted. Implementation in progress.

This document is the engineering specification derived from the accepted project
proposal (`proposals/HiddenStates-Proposal.pdf`) and the course project
guidelines (`Project_Guidelines.pdf`). The proposal is the scientific argument;
this file is the contract for what gets built, what counts as done, and when.

---

## 1. Problem Statement

Vision-language models (VLMs) that read charts and diagrams fail in two
mechanically different ways:

- **Structural misreading.** The model picked the wrong bar, misjudged an axis
  tick, or traced the wrong edge. The correct answer was present in the image,
  so a second, more careful look may recover it.
- **Fabrication.** The model stated a value or label that appears nowhere in the
  figure. Looking again will not help; the answer should be withheld.

Existing reliability tooling collapses both into a single "do not trust" verdict.
Abstaining on everything discards recoverable answers; re-querying everything
spends compute on answers that were never grounded.

**Core question.** Is the distinction visible in the model's internal state
*before* generation, and is knowing it worth anything?

## 2. Hypothesis

**Type Separability Hypothesis.** The pre-generation internal state preceding a
structural misreading is linearly distinguishable from the state preceding a
fabrication, because the former is a precision-limited perceptual failure while
the latter reflects generation ungrounded in the image.

This is under test, not assumed. Charts pass through the same vision encoder as
photographs, and misreading a bar height may be a higher-precision instance of
the same perceptual failure rather than a distinct mechanism. The experimental
design (Section 6) is arranged so that a collapse of the two classes is a
reportable finding, not a dead end.

## 3. Scope

### In scope
- Activation caching from two frozen open VLMs on structured-image QA.
- Three linear probes (binary, structural one-vs-rest, fabrication one-vs-rest).
- A surface-feature control baseline and residualisation analysis.
- A failure-type x repair cross-intervention matrix computed from ground-truth
  labels, independent of any probe.
- A type-aware controller evaluated against a binary selective-prediction
  baseline and a label-aware oracle.
- A synthetic data generator for bar charts and node-link diagrams.

### Out of scope
- Any weight update to a VLM. All VLMs stay frozen. The only trained components
  are the probes and the controller.
- Finer-grained taxonomies (localisation / association / scale subtypes). Two
  classes only. See Section 5.3.
- Natural-photograph hallucination benchmarks (POPE and similar). Structured
  images only.
- Attention-head-level mechanistic interventions (VIB-probe style). Cited as
  related work, not reimplemented.

## 4. System Components

### 4.1 Models and activation extraction
- **Primary models:** Qwen2.5-VL-7B, LLaVA-NeXT. Both frozen.
- **Hardware:** Ada for development and probes; H100 (via Sanjith) for the heavy
  inference jobs. Each model is pinned to one pool for the whole project, since
  differing kernels can change the generated answer and would decouple cached
  activations from the answer that was actually labelled.
- **Stretch goal:** ChartGemma. Only attempted if the two primary models are
  fully through the pipeline.
- For each `(figure, question)` pair, cache activations at a fixed subsampled set
  of layers spanning early, middle and late depths, at both **vision-token** and
  **query-token** positions. HALP (Kogilathota et al., 2026) reports that the
  most informative representation family varies by architecture, with
  Qwen2.5-VL-7B best served by visual-only features; Liu et al. (2026) report
  the two pathways occupy different depths. Both facts motivate spanning depth
  and position rather than fixing one.
- **Binding constraint is storage, not compute**, but less so than assumed.
  Pooling the vision tokens brings a full-depth cache to roughly 35 GiB across
  both models, so activations are cached at **every** layer rather than a
  subsampled set, falling back to eight layers only if shared storage quota
  requires it. Caching per-token vision activations remains out of the question
  at roughly 4.6 TiB. The layer index set is recorded in the run config so
  caches are reproducible. See `configs/activations.yaml`.
- **Ada's `/scratch` is node-local, not shared.** What a job writes on one node
  is invisible to a job that lands on another. It is fast working space for the
  duration of a job, not the cache's home. Every job stages its activations back
  to shared storage before exiting, or the cache ends up scattered across
  whichever nodes the scheduler happened to pick.

### 4.2 Probes
Three logistic regressions with L2 regularisation, trained on cached activations:

| Probe | Task | Role |
| --- | --- | --- |
| `binary` | correct vs. incorrect | Sanity baseline. Replicates HALP on structured images. If this fails, nothing downstream is interpretable. |
| `structural` | structural vs. rest | Type probe. |
| `fabrication` | fabrication vs. rest | Type probe. |

Per-layer AUROC is reported for each. **Layer selection happens on validation
data only.**

### 4.3 Cross-intervention matrix
This experiment **does not use the probes at all.** Taking only incorrect
answers carrying ground-truth failure labels, apply every repair to every failure
type and record recovery:

- `I_0`: no intervention. Gives the baseline recovery rate.
- `I_crop`: re-query at higher resolution on a **question-conditioned** crop.
- `I_verify`: re-query with a prompt demanding the model name the figure region
  supporting its answer.
- `I_abstain`: withhold the answer.

Each cell reports `P(correct | failure type i, intervention j)`. Abstention is
scored as **risk avoided**, not accuracy gained.

**Design constraint.** The crop must be conditioned on the question, never on the
model's own attention. Cropping to where a model attended when it misread would
reproduce the error.

**Design question.** Does the matrix have diagonal structure? If cropping rescues
fabrications as often as misreadings, the taxonomy earns nothing however well the
probes perform, and we would rather learn that directly than infer it from probe
accuracy.

### 4.4 Type-aware controller
Given a matrix with structure, train a controller mapping cached activations to
an intervention choice. Three settings are compared:

1. **Oracle:** sees the true failure type. This is the ceiling on what
   type-awareness can buy.
2. **Learned controller:** reads activations only.
3. **Binary policy:** abstains on everything the binary probe distrusts.

## 5. Data

### 5.1 Naturalistic arm
Run both VLMs over **ChartQA** (Masry et al., 2022), collect incorrect answers,
and label each with an LLM judge given the figure, question, gold answer and
model answer, following the methodology (not the taxonomy) of Ashury-Tahan et
al. (2026).

**Agreement gate.** Roughly 200 items are independently annotated by two team
members. Acceptance threshold is **kappa > 0.6**. Below it: simplify the category
definitions, adjudicate disagreements, and re-run. We do not proceed on
unreliable labels.

### 5.2 Synthetic arm
Programmatically generate bar charts and node-link diagrams with known ground
truth. This arm exists for three reasons:

1. Labels are correct by construction rather than inferred by a judge.
2. Matched item pairs can hold visual precision fixed while varying relational
   depth, and vice versa, separating the two factors by design instead of post
   hoc.
3. Class balance is directly controllable. Fabrication errors are expected to be
   scarce in natural data, and the cross-intervention matrix needs adequate
   counts in every cell.

The generator is built **in parallel** with the ChartQA runs, not after them.

### 5.3 Taxonomy
Two error classes: `structural` and `fabrication`. Deliberately not finer.
Splitting further would multiply annotation burden, degrade judge agreement, and
leave too few examples per class to support reliable probing.

### 5.4 Splits
Splits are constructed so that **no figure appears in both train and test.**
Layer and hyperparameter selection uses validation data only.

## 6. Experiments and Success Criteria

| ID | Experiment | Done when |
| --- | --- | --- |
| **E1** | **Probe quality.** Per-layer AUROC for all three probes. | Binary probe AUROC falls within HALP's reported range as a sanity condition. Per-layer curves produced for all three probes on both models. |
| **E2** | **Probe-surface gap.** Train a classifier on surface features alone: answer present in figure text, answer type, answer length, question template, figure type. Report the margin by which the activation probe exceeds it, plus probe accuracy after residualising these features out. | Both the surface baseline and the residualised probe accuracy are reported alongside every raw AUROC. **The gap, not the raw AUROC, is the evidence.** |
| **E3** | **Separability.** Each type probe is trained on its own class and tested for cross-transfer. | Cross-transfer matrix reported. Strong cross-transfer indicates one signal wearing two labels, and is reported as such. |
| **E4** | **Cross-intervention matrix.** Full failure-type x repair recovery matrix from Section 4.3. | Every cell populated with bootstrap confidence intervals. |
| **E5** | **Utility.** Risk-coverage curves and AURC for the plain VLM, binary selective prediction, the learned controller and the oracle, at matched coverage. | All four curves at matched coverage, plus an ablation isolating whether re-querying contributes anything beyond abstention. |

**No result is reported without its control.** E1 without E2 is not a result.

### 6.1 Outcome branches
E4 is independent of E1 to E3, which is what makes the project robust. Every
branch below is a publishable outcome and all are written up the same way:

- **Probes separate AND matrix has diagonal structure:** a working type-aware
  policy. Proceed to E5 in full.
- **Matrix has structure BUT probes fail:** different failures need different
  repairs, but cannot be anticipated from internal state. A negative result about
  predictability, not about the taxonomy.
- **Matrix is flat:** the distinction does not pay. Report that alongside the
  binary detection results, which would still be the first pre-generation
  detection numbers for structured images.

### 6.2 Go / no-go checkpoint
**30 September 2026.** By then we know whether the probes separate the classes
and whether the surface baseline explains them away. That decides whether the
second half pursues the controller (E5) or documents the negative result.

## 7. Deliverables and Deadlines

From the course guidelines:

| Deliverable | Deadline | Requirements |
| --- | --- | --- |
| Final Proposal | 20 Aug 2026 | Submitted. `HiddenStates-Proposal.pdf`. |
| **Mid Submission** | **30 Sep 2026** | `HiddenStates-Mid.zip`. 7-8 pages excluding references, optional 1 appendix page. Formal refined problem statement, progress, concrete timeline for the remainder, comprehensive literature review. Repo public or TA mentors granted access. Graded on write-up, code and viva. Slides optional. |
| **Final Submission** | **31 Oct 2026** | `HiddenStates-Final.zip`. At most 8 pages excluding references, optional 2 appendix pages. Structured as a research paper. Every claim backed by prior or empirical results with analysis. Code as a ZIP in the submission. Presentation PDF for the viva. |

### Standing constraints
- **ACL Style Guide** for all write-ups.
- **AI must not be used to generate the reports.** AI use for implementation and
  code is permitted. This constraint is binding on every write-up deliverable.
- **Every team member submits** each deliverable individually.
- No extensions. Deadlines are hard.
- Any HuggingFace or WandB projects must be linked from the repository README.
- Challenges, limitations and future work go in a separate section immediately
  before the references in the final report, and that section must contain
  nothing else. Any project result placed there will be graded as future work.

## 8. Timeline

| Period | Work | Status |
| --- | --- | --- |
| Aug 28 - Sep 10 | Ada setup; VLM runs on ChartQA | In progress |
| Sep 11 - Sep 21 | Judge labels; kappa check; activation caching | Pending |
| Sep 22 - Sep 30 | Probes; surface control (E1, E2, E3) | Pending |
| Oct 1 - Oct 15 | Cross-intervention matrix; oracle (E4) | Pending |
| Oct 16 - Oct 31 | Controller; risk-coverage; write-up (E5) | Pending |

The synthetic generator runs in parallel with the ChartQA runs, since under five
weeks separate proposal acceptance from the mid-submission.

## 9. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Low judge agreement (kappa <= 0.6) | The synthetic arm needs no judge. Its labels are correct by construction. |
| Fabrication scarcity in natural data | Ask about quantities outside a figure's range; control class balance directly in the synthetic arm. |
| Signal is diagnostic but not causal (the Yuan et al. 2026 failure mode) | E4 stands alone without any probe. It is not downstream of E1 to E3. |
| Probe result is a format or surface confound (the Sahoo et al. 2026 failure mode) | E2 is mandatory, not optional. No AUROC is reported without its surface baseline and residualised counterpart. |
| Activation storage | Pool the vision tokens, which is what makes a full-depth cache affordable; use node-local `/scratch` as working space and stage back to shared storage per job; record the layer set in the run config. |
| Compute | Not binding. 7B forward passes fit on Ada, probes take minutes on CPU, and Sanjith has H100 access for the heavy inference jobs. Real costs are activation storage and judge API calls. |

## 10. Repository Layout

Target structure. Directories are created as the corresponding stage lands.

```
type-aware-vlm-failures/
  SPEC.md                 this file
  README.md               setup, how to run, HF/WandB links (required by guidelines)
  configs/                run configs: model, layer set, token positions, splits
  src/
    extract/              VLM inference and activation caching
    label/                LLM judge pipeline and annotation agreement (kappa)
    synth/                synthetic bar-chart and node-link generator
    probes/               binary, structural, fabrication probes
    surface/              E2 surface-feature baseline and residualisation
    intervene/            I_0, I_crop, I_verify, I_abstain and the E4 matrix
    controller/           type-aware controller, oracle, binary policy
  scripts/                Ada / SLURM job submission
  results/                metrics, figures, matrices (small artefacts only)
  paper/                  ACL-style LaTeX for mid and final write-ups
```

Activation caches live on shared cluster storage, staged there from node-local
`/scratch` at the end of each job. They are never committed.

## 11. Definition of Done

The project is complete when:

1. E1 to E5 have each either produced their reported artefact or been explicitly
   closed by an outcome branch in Section 6.1, with the reason recorded.
2. Every reported probe number is accompanied by its surface baseline (E2).
3. The cross-intervention matrix has bootstrap confidence intervals in every cell.
4. Splits are verified to share no figure between train and test.
5. The repository is public or TA mentors have access, and the README links any
   HuggingFace or WandB projects.
6. `HiddenStates-Final.zip` is submitted by every team member on 31 Oct 2026,
   containing the ACL-style write-up, the code archive and the presentation PDF.

## References

Ashury-Tahan et al. 2026. ErrorMap and ErrorAtlas: Charting the failure landscape
of large language models. arXiv:2601.15812.

Iyengar et al. 2026. DRAGON: A benchmark for evidence-grounded visual reasoning
over diagrams. arXiv:2604.25231.

Kogilathota et al. 2026. HALP: Detecting hallucinations in vision-language models
without generating a single token. EACL 2026, pages 6067-6085.

Li et al. 2026. Visual self-refine: A pixel-guided paradigm for accurate chart
parsing. ICLR 2026.

Liu et al. 2026. Dual-pathway circuits of object hallucination in vision-language
models. arXiv:2605.13156.

Masry et al. 2022. ChartQA: A benchmark for question answering about charts with
visual and logical reasoning. Findings of ACL 2022.

Rani et al. 2025. RADAR: A reasoning-guided attribution framework for explainable
visual data analysis. arXiv:2508.16850.

Sahoo et al. 2026. Linear probes detect task format, not reasoning mode in
language model hidden states. arXiv:2606.02907.

Yuan et al. 2026. Hidden error awareness in chain-of-thought reasoning: The
signal is diagnostic, not causal. arXiv:2605.09502.

Zhang et al. 2026. VIB-probe: Detecting and mitigating hallucinations in
vision-language models via variational information bottleneck. ACL 2026, pages
23509-23521.
