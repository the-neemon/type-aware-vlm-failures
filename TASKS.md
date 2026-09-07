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
| **A. Infra and inference** | Ada, SLURM, envs, model runs, activation caching | Yash More |
| **B. Labelling** | Annotation tooling and protocol, kappa, taxonomy definitions | Shrish Kadam |
| **C. Synthetic** | Generator, matched pairs, ground-truth labels | Sanjith Ganapathi |
| **D. Analysis** | Probes, surface control, matrix, controller, plots | Naman Singhal |

Provisional, swap freely at the first meeting. Every member needs enough context
on workstream B to annotate, since the kappa check needs two independent
annotators, and the judge-versus-human check in P2.5 needs a third opinion to
adjudicate.

### 1.1 Compute resources

Two pools, and they are not interchangeable.

**Ada is the only pool.** Sanjith's H100 access belongs to his own research and
is not available to this project. Plan accordingly: there is no fast fallback if
Ada queues get long, so start the long inference jobs early rather than in the
week before a deadline.

**The consistency requirement survives losing the H100, and Ada makes it
harder.** A model's answers and its activations must come from the same
hardware, or P4.4's coupling breaks silently and nothing downstream flags it.
Ada's nodes carry mixed GPU types, so "run it on Ada" is not a pin. Every
inference and caching job names one GPU type in its SLURM constraint, that type
is frozen for the project, and it is recorded in the run config next to the
library versions (Section 4.5). Decision 9.

**Compute is moderate, not trivial.** Roughly 7,000 forward passes for
inference and error collection (about 3,500 questions per model), plus caching,
plus re-queries for the intervention matrix, most of which now fall on the
synthetic arm. That fits on Ada comfortably, but still not in a one-hour
walltime slot, so see P1.7 on resumability.

### 1.2 Daily plans

Today's tasks live in [TODAY.md](TODAY.md), rewritten each working day.
The phase breakdown below is the backlog those tasks are drawn from.

--- | --- | --- |
| **Yash** | Ada, environment, weights, smoke test, token-count check (P0.1, P0.2, P0.3, P0.5, P0.10) | One Qwen2.5-VL-7B answer to one ChartQA image from a committed `scripts/smoke.sbatch`, plus a pass or fail on `scripts/verify_vision_tokens.py` |
| **Naman** | Repo skeleton, then the storage budget and caching schema (P0.6, P0.7, P0.8, P0.4) | `configs/activations.yaml` plus the arithmetic behind it, as a proposal for the evening sync |
| **Shrish** | ChartQA ingest, relaxed accuracy, figure-level splits (P1.1, P1.6, and the P5.4 assertion) | `src/eval/relaxed_accuracy.py` with boundary tests passing, and figure-versus-question counts for every split |
| **Sanjith** | Synthetic bar chart generator (P3.1), taxonomy definitions draft (P2.1) | Twenty generated charts with a manifest, and a first draft of the two class definitions |

**Detail per person.**

*Yash.* Confirm all four of us can submit an Ada job and write to `/scratch`
before doing anything else, since a missing account is a multi-day fix. Then map
the filesystem, because `/scratch` is node-local and cannot hold anything the
whole team shares: find the shared path we are actually entitled to, and report
its quota. That number decides whether the checkpoints can be downloaded once or
have to be replicated per node. Pin `transformers` and `qwen-vl-utils` in `requirements.txt` and
record the exact versions, because activations differ silently across versions.
In the SLURM template, request walltime explicitly (the default is one hour),
constrain the GPU type since nodes are mixed, and exclude `gnode077`. The day is
successful if one image and one question produce one answer string. Do not batch,
do not evaluate, do not tune the prompt yet.

Once that works, run P0.10: `python3 scripts/verify_vision_tokens.py` against
five real ChartQA figures. It takes a minute and it is the check that decides
whether tonight's storage numbers are real or an artefact of a reimplemented
`smart_resize`. If it fails, say so before the sync rather than after.

*Naman.* Skeleton first, then the critical-path decision. Read
`num_hidden_layers` and `hidden_size` from each model's `config.json` rather than
trusting the numbers quoted in Section 4.1 of this file; verifying them is the
task. Then run the processor on three real ChartQA figures at the resolution we
intend to use and count the actual vision tokens after patch merging, since that
is the term that decides whether the cache is 2 GB or 1.8 TB. Produce the
arithmetic for pooled and unpooled at a candidate layer set (for 28 layers,
something like 2, 6, 10, 14, 18, 22, 26, 27 spans early, middle and late), and
bring a recommendation rather than options. This needs no GPU.

*Shrish.* Download ChartQA and count figures against questions per split, which
tells us how much figure-level splitting will cost us in effective sample size.
Implement relaxed accuracy with 5 percent numeric tolerance and exact match on
strings, and unit-test it exactly at the tolerance boundary in both directions.
Getting this wrong marks correct answers as errors, and those mislabelled items
land in the structural class and poison every probe downstream, so it is worth a
full day. Write the manifest schema for `(figure_id, question, gold, prediction,
correct)` while you are in the data, since both caching and labelling key off it.
Add the assertion that no `figure_id` appears in two splits.

*Sanjith.* Start with bar charts only, node-link diagrams can wait. Make bar
height separation, tick density and axis range controllable parameters from the
first version, since those are what P3.4's matched pairs will vary. Emit a
manifest alongside the images carrying ground truth and the split assignment, so
the split can never drift from the data. Separately, draft the structural and
fabrication definitions with three worked edge cases each, and include the
explicit "neither / ambiguous" option; that draft is what Shrish's judge prompt
gets built on next week.

**Evening sync, 30 minutes.** One agenda item that matters: ratify or reject
Naman's pooling recommendation (open decision 1). It is close to irreversible
once caching starts, so it gets decided by the team rather than by whoever writes
the caching code. Everything else is a status round.

---

## 2. Phase Breakdown

### P0. Infrastructure (2 Sep to 8 Sep, 1 week)

- [x] **P0.1** Confirm Ada access for all four members. Verify each can submit a
      job and write to `/scratch`.
      All four hold a SLURM association under account `research`; usernames,
      uids and QOS are in `results/ada_filesystem.md` Section 1. Sanjith is on
      QOS `low` rather than `medium`, so his jobs queue behind ours.
      **The other three still have to run the one-line check in that section
      themselves.** An association is what gates submission, but I cannot
      submit a job as another user, so this is ticked on the evidence I could
      actually gather.
- [x] **P0.2** Pin the environment. Qwen2.5-VL needs a recent `transformers` plus
      `qwen-vl-utils`; LLaVA-NeXT has its own processor path. Lock versions in
      `requirements.txt` and record the exact commit. Version drift between team
      members will silently change activations.
      Done. `requirements.txt` carries exact pins, built by
      `scripts/setup_env.sh` and verified on a 2080 Ti node: torch 2.8.0+cu126,
      transformers 4.57.6, qwen-vl-utils 0.0.14, accelerate 1.14.0,
      datasets 5.0.1, numpy 2.2.6, pillow 12.3.0, Python 3.10.12.
      transformers is deliberately the last of the 4.x line rather than 5.x,
      because Qwen2.5-VL and qwen-vl-utils were written against the 4.x
      processor and generation APIs.
      The venv lives in `$HOME` (6.2 GiB) and **must be built from inside a
      job**: the head node runs an older glibc than the compute nodes, so a
      venv built there will not import once a job picks it up.
- [x] **P0.3** **Map the filesystem before downloading anything.** `/scratch` on
      Ada is node-local: a job on one node cannot see what a job on another node
      wrote there. Report (a) the shared path available to us and its quota,
      (b) home quota, (c) whether `/scratch` is ever purged and on what cycle.
      Then decide where `HF_HOME` points. Two 7B checkpoints are roughly 32 GB
      combined, so if no shared location holds them, they get replicated per
      node and jobs must be pinned to a fixed node set to avoid re-downloading
      16 GB every time the scheduler moves us. Resolves open decision 8.
      Measured 2 Sep; full report in `results/ada_filesystem.md`.
      (a) The only shared, writable, durable path is `$HOME`, 30 GiB per user.
      `/share1` holds 100 GiB but sits on the head node, is not mounted on any
      compute node, and compute nodes cannot ssh to the head node, so no job
      can reach it.
      (b) Home quota 30 GiB soft, 31 GiB hard.
      (c) **`/scratch` is purged.** It is bind-mounted to the same directory as
      `/tmp` (same device and inode), and `tmpreaper` runs daily over `/tmp/.`
      at the 7-day default; the oldest surviving entry observed was 5 days old.
      So `HF_HOME=/scratch/vlm-failures/hf` is never set up once:
      `stage_in_model` in `scripts/ada_env.sh` repairs it on whichever node a
      job lands on, a no-op on a warm node.
- [ ] **P0.9** Job scripts stage results back. Every job that writes activations
      copies them from node-local `/scratch` to the shared path before exiting,
      and the script fails loudly if the copy fails. Without this the cache
      silently fragments across nodes and is unusable for training a probe.
- [x] **P0.10** **Verify the vision-token counts against the real processor.**
      The storage budget rests on counts that `src/extract/storage_budget.py`
      derives analytically, by reimplementing `smart_resize` from the config.
      Nothing has confirmed that reimplementation matches what
      `Qwen2VLImageProcessor` actually does, and an off-by-one in the merge or a
      different rounding rule shifts the entire budget table. Run
      `python3 scripts/verify_vision_tokens.py <figures>` on at least five real
      ChartQA figures of differing sizes, with `--max-pixels` matching
      `configs/activations.yaml`. It exits non-zero on any mismatch.
      - Pass: report the observed token range so the planning number in
        `configs/activations.yaml` can be replaced with a measured one.
      - Fail: `storage_budget.py` is wrong. Fix it, regenerate
        `results/storage_budget.md`, and recheck the schema before any caching
        starts.
      Depends on P0.2 and P0.3, and on a handful of ChartQA figures from
      Shrish's P1.1 ingest (or download a few directly, five is enough).
      **PASS, 7 Sep.** All five figures matched exactly, so the analytic
      `smart_resize` in `src/extract/storage_budget.py` models the real
      processor correctly and the budget table stands. Observed range **180 to
      630 tokens**; 800x557 is the common ChartQA size and gives 580. The old
      planning number of 1044 was a hypothetical 800x1000 figure and has been
      replaced with the measured 580. This does not move the budget either way:
      pooled cost per item is vectors x hidden x layers and never references the
      token count; the count only fed the rejected unpooled schema.
      Full report in `results/p0_10_vision_tokens.md`.
      **Secondary finding, and the more consequential one.** transformers 4.57
      loads the *fast* image processor by default and warns it "may produce
      slightly different outputs". It does: preprocessed pixels differ from the
      slow processor on every figure tested, worst absolute difference 0.030 on
      normalised values. Token counts are unaffected, so P0.10 stands, but the
      pixels feeding the encoder are not, and that is what we cache. Two of us
      on different defaults would cache different activations from the same
      figure and neither would notice. `processor_use_fast: true` is now frozen
      in `configs/activations.yaml` next to `gpu_type`, and passed explicitly in
      both the inference and verification paths.
- [x] **P0.4** **Decide the caching schema and compute the storage budget before
      writing any cache.** This is the single most consequential infra decision
      in the project. See Section 4.1 for the arithmetic.
      Budget computed in `results/storage_budget.md` from
      `src/extract/storage_budget.py`; schema proposed in
      `configs/activations.yaml`. Ratification is open decision 1, Section 7.
      One check still outstanding, now owned by workstream A as P0.10: the
      vision-token counts are analytic and unverified against the real
      `Qwen2VLImageProcessor`.
- [x] **P0.5** SLURM job template. Request explicit walltime; the cluster default
      is 1 hour and a full ChartQA pass will not finish in it. Pin GPU type in the
      constraint, since nodes are mixed and a job that lands on the wrong card
      will OOM or run slow. Avoid `gnode077` (known dead GPU).
      `scripts/ada_env.sh` is the shared preamble; `scripts/smoke.sbatch` and
      `scripts/verify_tokens.sbatch` are the two templates. Explicit `--time`,
      `--constraint=2080ti`, `--exclude=gnode077`, and `--gres=gpu:2` because
      16.6 GiB of fp16 weights does not fit one 11 GiB card. `require_gpu_type`
      re-checks the card at runtime rather than trusting the scheduler, and
      `stage_out` implements P0.9 by failing the job loudly if output does not
      leave node-local disk.
      Two Ada facts that bit and are now encoded: the GPU is Turing (sm_75), so
      there are no bf16 tensor cores and no FlashAttention-2, and the project
      therefore runs fp16 with `sdpa`; and QOS `medium` caps a user at 4 GPUs
      and 40 CPUs in total, so a held interactive session leaves batch jobs
      pending on `QOSMaxGRESPerUser`.
- [x] **P0.6** Repo skeleton per SPEC Section 10: `configs/`, `src/`, `scripts/`,
      `results/`, `paper/`. Add `.gitignore` for caches, checkpoints and `*.npz`.
- [x] **P0.7** README stub. Guidelines require HuggingFace and WandB links live
      here. Add the section now even if empty.
- [ ] **P0.8** Grant TA mentor access to the repo, or make it public.
      **Deferred by team decision on 2 September.** Note the binding deadline is
      the **mid** submission (30 September), not the final one: the guidelines
      require the repo to be public or TA-accessible at that point. Whoever
      picks this up in late September, it is a two-minute change.

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
- [ ] **P1.5** Build an error pool of roughly **700 per model** (decision 3,
      revised). The original target of 1500 assumed an LLM judge could label
      cheaply at volume. Hand-labelling caps the naturalistic arm at about 1000
      items total, so inference only needs a pool comfortably larger than that
      to sample from. Test splits give roughly 500 errors per model; extend into
      train by about 1000 questions per model to clear 700. That is roughly 3500
      questions per model rather than 10,500.
- [ ] **P1.6** Store `(figure_id, question, gold, prediction, correct)` rows as
      the single source of truth that labelling and caching both key off.
- [ ] **P1.7** **Make inference resumable.** Even at the revised 3,500 questions
      per model, a run will not finish in one walltime slot. Jobs write results incrementally and
      skip items already present on restart, keyed by `(model, figure_id,
      question)`. Without this, every queue eviction costs the whole run.

### P2. Labelling (8 Sep to 21 Sep, 2 weeks, overlaps P1)

- [ ] **P2.1** Write the two-class definitions with worked edge cases. Include an
      explicit **"neither / ambiguous"** escape hatch so annotators and the judge
      are never forced to pick. Ambiguous items are dropped, not coerced.
- [x] **P2.2** Build the **annotation tool**, not a judge pipeline (decision 2:
      no budget for paid APIs). A minimal local interface is enough: show the
      figure, question, gold answer and model answer, take one of three keys,
      capture a one-line rationale, write to disk, next item. Annotators must
      not see the model identity or each other's labels. Budget half a day; the
      tool is not the deliverable, the labels are.
- [ ] **P2.3** Pilot on 50 items with **all four** annotating. Compare, argue
      about the disagreements, revise the rubric. Expect at least two rounds.
      This is where the taxonomy actually gets defined; the rubric written in
      P2.1 is a draft until it survives this.
- [ ] **P2.4** **Kappa gate.** Two team members independently annotate roughly
      200 items. Compute Cohen's kappa. Threshold is 0.6.
      - Pass: proceed.
      - Fail: simplify definitions, adjudicate, re-annotate. Do not proceed on
        unreliable labels. Budget a full week for a failure.
- [ ] **P2.5** ~~Judge-versus-human agreement.~~ **Moot under decision 2**: with
      no judge, inter-human kappa is the whole validation story. This is the one
      genuine upside of losing the API budget, and the paper should say so
      plainly rather than presenting hand-labelling as a limitation only.
- [ ] **P2.6** Annotate the main set: 1000 items across both models, roughly
      250 each, after the rubric is frozen. Record the one-line rationale per
      item for the appendix. Do this in two sittings rather than one; agreement
      degrades with fatigue and that degradation is invisible in the output.

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
- [ ] **P3.6** Generate at scale with class balance set deliberately. **400 items
      per failure type is a floor, not a target.** The E4 power simulation
      (`results/e4-preregistration.md`) puts the minimum detectable effect at 400
      per cell at roughly 12 percentage points, which is large; at a true effect
      of 5 points that design finds nothing 73 percent of the time. Synthetic
      items are free and the synthetic arm carries E4's cell counts, so generate
      well past the floor: **target 800 errors per failure type per model**,
      which buys 0.94 power at a 10-point effect. Note this is 800 *errors*, not
      800 items; at a plausible 30 percent synthetic error rate that is roughly
      2,700 items per type per model. E4 needs inference but not activations, so
      this costs GPU time and nothing in the activation cache.

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
- [x] **P5.3** Layer selection on **validation only**. Write the selection code so
      test data is not even loaded during selection.
      Done in `src/probes/sweep.py`: `sweep_layers` takes no test parameter at
      all, so selecting on test requires deliberately misusing the API rather
      than forgetting a rule. A test asserts the signature stays that way.
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
      `cluster_bootstrap_ci` in `src/common/linear.py` does this; see Section 5.5
      for when the clustering actually changes the interval and when it does not.
- [x] **P6.7** Test for diagonal structure explicitly. State the test in advance
      rather than eyeballing the matrix.
      Pre-registered on 2 September in `results/e4-preregistration.md`, before
      any intervention has been run. Primary test is a directional difference in
      differences on `I_crop` with a cluster-bootstrapped interval;
      `argmax_flip_stability` is reported beside it, never instead of it.
      `I_abstain` is excluded by design and the code raises rather than silently
      returning a number for it. Implementation in `src/intervene/matrix.py`.

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

**The team is not available uniformly.** Midsems run 21 to 28 September and the
mid submission is due 30 September, which means the write-up has to be finished
*before* the exams start, not after them.

| Window | Days | Availability | Focus |
| --- | --- | --- | --- |
| 3 to 10 Sep | 8 | **Full** | Close P0; inference complete; synthetic arm complete; rubric frozen and piloted; caching implemented |
| 11 to 14 Sep | 4 | Low, travelling home | **Annotation round 1**; start the literature review. Caching jobs run unattended |
| 15 to 20 Sep | 6 | Low, exam preparation | Kappa check; probes; literature review continues |
| 21 to 24 Sep | 4 | **Zero, midsems** | Nothing |
| 25 to 30 Sep | 6 | **Full** | Results sections, assembly, submit |
| 1 to 31 Oct | 31 | Full | E4 matrix, E5 controller, final write-up |

### 3.1 Hard internal deadlines

These are not the same as the course deadlines, and missing one costs more than
it looks like it should.

| By | What | Why it is hard |
| --- | --- | --- |
| **10 Sep** | Inference finished on both models | Annotation is the only task that works well in the low-availability travel week. Without an error set by the 10th, that week is wasted and the annotation has nowhere left to go. |
| **14 Sep** | Annotation round 1 complete | Everything typed depends on labels: E1's type probes, E3, E4's rows, E5's oracle. |
| **20 Sep** | Experiments done and the write-up drafted | Only 4 exam days follow, but they are followed by 6 working days that have to hold the entire results write-up. Arriving at the 25th with experiments still running is what turns a comfortable window into a scramble. |

### 3.2 Put the literature review in the low-availability weeks

It is the one substantial piece of the write-up that does not depend on a single
result. It is reading and writing, it works in a bedroom with no cluster access,
and it survives being done in 40-minute pieces. The guidelines grade it
explicitly and it is the section teams most often under-weight.

Drafting it across 11 to 20 September alongside annotation means the 25th to
30th holds only the results sections, which are the parts that genuinely cannot
be written earlier. Write it by hand: the course forbids AI-generated reports,
and that applies to the literature review as much as anything else.

### 3.3 What the mid submission realistically contains

Scope honestly rather than promising E1 to E5. On the effective time available,
which is roughly 25 to 30 hours per person before the exams, a credible mid
submission is:

- P0 to P4 complete: pipeline, data, labels, activations cached
- **E1**: the binary probe checked against HALP's range as the sanity condition,
  and the two type probes
- **E2**: the surface baseline, reported next to every probe number
- **E3**: cross-transfer between the type probes
- Literature review, refined problem statement, and the remaining timeline
- E4 and E5 are October work and are described as planned, not attempted

The guidelines ask for progress and a concrete plan, not finished results. A
smaller set of honest, controlled numbers reads better than a larger set of
uncontrolled ones, and E2 is what makes them controlled.

### 3.4 Working independently

Nobody is in the same room. Two consequences worth stating.

Tasks are assigned so that each person works inside their own directory
(`src/extract`, `src/label`, `src/synth`, `src/probes`), which keeps merge
conflicts close to zero. Pull before starting and push when you stop, even if
the work is unfinished; a branch nobody can see is a branch nobody can unblock.

When a task is blocked, say so in the repo rather than waiting to be asked.
Adding a line to the open decisions table in Section 7 is enough.

**Size work for agent-assisted speed.** Everyone here works with a coding
agent, so a task that reads like an hour of hand-written work lands in about ten
minutes. Day plans should hand each person a coherent chunk of a phase rather
than a single task, or the schedule looks far tighter than it is and the scoping
decisions that follow from it come out wrong.

## 4. Things To Be Careful About: Infrastructure

### 4.1 Activation storage will explode if you cache raw vision tokens

**Measured, not estimated.** See `results/storage_budget.md`, regenerable with
`python3 src/extract/storage_budget.py`. Verified dimensions: Qwen2.5-VL-7B is
3584 wide over 28 layers, LLaVA-NeXT-7B is 4096 over 32. A typical ChartQA
figure costs Qwen 1044 vision tokens; LLaVA-NeXT is fixed at 2928.

Across both models at 20k items, fp16:

| Schema | 8 layers | All layers |
| --- | --- | --- |
| Pooled, 4 vectors per layer | 9.2 GiB | 34.5 GiB |
| Unpooled vision tokens | 4.6 TiB | 18 TiB |

Two consequences. Unpooled is ruled out by three orders of magnitude, as
expected. But pooled is so cheap that **subsampling layers saves nothing worth
having**: full depth costs 3.8x on a 9 GiB base, and buys away the risk of
having picked the wrong layers, which matters because Liu et al. place the two
pathways at different depths. This reverses the SPEC's assumption that storage
forces subsampling.

Store fp16, never fp32. If a later experiment genuinely needs per-token
activations, re-cache at most a few hundred items rather than changing the
schema.

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
cannot hold it. Avoid `gnode077`.

**`/scratch` is node-local, not shared.** It persists across jobs *on the same
node*, which makes it good working space and a good weights cache, but a job
that lands elsewhere sees none of it. Two consequences. Model weights cannot be
downloaded once and shared through it, so either they live on shared storage or
they are replicated per node and jobs are pinned to a fixed node set. And
activations written there must be staged back to shared storage before the job
exits (P0.9), or the cache fragments across whichever nodes the scheduler picked
and no probe can be trained on it.

This is another argument for pooling: 35 GiB of pooled activations will fit on
shared storage, whereas 4.6 TiB of unpooled activations would have had nowhere
to live at all.

### 4.5 Reproducibility hygiene

**The seed for this project is 42**, everywhere, including new code. Written
down because it is worth not relitigating.

Fix seeds, pin library versions, and write the full run config next to every
results file. Two team members running "the same" job on different transformers
versions will produce different activations and neither will notice.

---

## 5. Things To Be Careful About: Methodology

### 5.1 Every label is now a human label, which changes what kappa means

Under decision 2 there is no judge, so kappa between annotators is no longer a
proxy for anything: it *is* the label-validity evidence, covering the whole
dataset rather than a 200-item sample of it. That is a genuine strengthening and
the paper should say so, because the obvious reading of "we could not afford an
LLM judge" is that the labels got worse.

What it costs instead is sample size, and the risks move accordingly. Watch for
annotator drift, where the rubric is applied differently in week three than in
week one; re-annotate a 50-item slice from the first sitting at the end and
check that the labels still agree with themselves. Watch for one annotator's
idiosyncratic reading propagating unchecked through the 750 items nobody else
sees. The double-annotated 200 is the only place either failure becomes visible,
so it must be sampled across the whole set and across sittings, not taken from
the front of the queue.

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

**Measured on 2 September, and the condition is narrower than the usual advice.**
Clustering widens the interval only when the label *and* the score are both
correlated within figure; if only one is, the two bootstraps agree to within
noise. Our case has both, so the guidance stands: a hard figure produces several
errors at once, and the activations behind one figure's questions share its
encoding. Both directions are pinned in
`src/probes/test_sweep.py::TestClusterBootstrap`, the second so that nobody
drops the grouping after checking it on a case where it genuinely makes no
difference. `cluster_bootstrap_ci` in `src/common/linear.py` does this.

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

### 5.8 Keep the naturalistic arm central, which just got harder

The synthetic arm is easier, cleaner and more fun to work on, and it will try to
eat the project. It is a control and a source of balanced classes. The claim is
about real chart reasoning. If the paper's headline numbers come from
synthetic data, the contribution shrinks accordingly.

**Decision 2 pushed weight toward synthetic and this is a real cost, not a
neutral reshuffle.** With no labelling budget, E4's cell counts come mostly from
generated figures, and the naturalistic arm supplies a smaller replication with
wider intervals. Two obligations follow. Report the naturalistic E4 alongside
the synthetic one even where its intervals are wide, because a matrix that has
diagonal structure only on synthetic data is a much weaker claim and the paper
must let a reader see that. And state the constraint in the limitations section:
the taxonomy's naturalistic evidence is bounded by what four people could label
by hand, not by anything about the taxonomy itself.

### 5.9 E2 only works on the naturalistic arm

Measured on 2 September (`results/e2-synthetic-is-tautological.md`): on
synthetic data the surface baseline reaches AUROC 1.000, because the question
template determines the label by construction and a fabrication item's answer
cannot appear in a figure that never contained the category. The probe-surface
gap can only be zero or negative there.

So E2 runs on the naturalistic arm or not at all, and hand-labelling now bounds
the power of the project's central control rather than just one experiment's
sample size. If something has to be cut, cut synthetic generation, not
naturalistic annotation.

### 5.10 Report the gap, never the raw AUROC alone

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
| 1 | Pooling strategy and cached layer set | 8 Sep | **Decided 2 Sep, rule revised same day.** Schema B, fp16, `layers: all`. At the planned 6k items the cache is 10.3 GiB against a 30 GiB quota, so it holds. **P0.10 passed 7 Sep**, so the token counts behind the budget are measured rather than assumed and the contingency is discharged. |
| 2 | How the naturalistic arm gets labelled | 10 Sep | **Decided 2 Sep.** No paid API. Hand-annotated, ~1000 items, ~250 each; synthetic arm carries E4. See `configs/labelling.yaml`. |
| 3 | ChartQA train split and error-pool size | 12 Sep | **Decided 2 Sep, revised same day.** Yes, but only ~1000 extra questions per model for a 700-error pool. Hand-labelling, not inference, is now the bottleneck. See P1.5. |
| 4 | Workstream ownership | Next meeting | Provisional, see Section 1 |
| 5 | `I_0` definition, with or without a naive re-ask control | 1 Oct | Open |
| 6 | `I_crop` conditioning mechanism | 1 Oct | Open |
| 7 | ChartGemma stretch goal: keep or drop | 30 Sep | Open |
| 8 | Shared storage path for weights and caches, given node-local `/scratch` | 8 Sep | **Closed 2 Sep by P0.3.** `durable_root=$HOME`, 30 GiB per user. `/share1` has 100 GiB but no compute node can reach it. Weights go to `HF_HOME=/scratch/vlm-failures/hf`, which is purged at 7 days and repaired on demand by `stage_in_model`. |
| 10 | Can probe training read `/share1` from the login node | 20 Sep | Open. Would unlock 100 GiB of cold archive; only matters if the item count grows. |
| 11 | Port `scripts/download_chartqa.ps1` to bash for Ada | 5 Sep | Open. PowerShell will not run on the cluster. |
| 9 | Which pool runs which model, fixed for the project | 14 Sep | **Decided 2 Sep, revised same day.** Ada only; the H100 is unavailable. One frozen GPU type, named in every job constraint. See Section 1.1. |
