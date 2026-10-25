# Outline: E4, E5 and discussion (for the team writing session)

Claims in order, one block per paragraph of `paper/final/sections/e4_e5_discussion.tex`.
Fragments, not report text: the team writes the prose. Every number is from the
file named; check it there before it goes in.

## E4, paragraph 1: setup

- Question: does the best repair depend on the failure type?
- Items: every value-question error, synthetic pilots 1 + 2. Qwen 631
  (46 misread, 91 made-up number, 494 "0"); LLaVA 1,015 (415 / 368 / 232).
  Source: `results/e4.md`.
- Four repairs: re-ask (sampled), upsample, question-conditioned crop, verify
  prompt. Same model, same decoding as the original answer.
- Recovery: misread → now correct by relaxed accuracy; missing bar → now refuses.
- Pre-registered on 2 Sep, before any run: DiD for crop =
  (crop gain on misreads) − (crop gain on missing-bar answers), predicted > 0;
  verify secondary; cluster bootstrap over figures.
  Source: `results/e4-preregistration.md`.

## E4, paragraph 2: result (Table e4)

- Prediction holds for both models: Qwen DiD +0.261 [+0.140, +0.389];
  LLaVA +0.152 [+0.119, +0.189].
- Best repair differs by type for Qwen: crop for misreads (0.26), verify for
  made-up numbers (0.14) and "0" (0.69). Argmax flip stability 0.99.
- LLaVA: crop fixes misreads (0.15); nothing fixes any missing-bar answer (0 of 600).

## E4, paragraph 3: limits (do not drop these)

- On a missing bar there is nothing to crop: crop shows the whole figure, so it
  equals upsample on all 1,185 missing-bar items. The crop's own gain on
  misreads is crop vs upsample: LLaVA +0.145 [+0.111, +0.181]; Qwen +0.130,
  interval touches 0 (46 misreads).
- LLaVA's "best repair differs by type" is a tie (all zeros), not evidence.
- Verify turns 342 of Qwen's 494 "0" answers into refusals: supports reading
  "0" as a refusal.
- Source: `results/e4.md`, "Read these before writing them up".

## E5, paragraph 1: setup

- Universe: all 1,200 value questions per model (pilots 1 + 2), right and wrong.
- Probe scores out of fold (5 figure folds): wrong-vs-right, missing-bar,
  misread. Source: `results/e5.md`, Provenance.
- Six policies (names as in Table e5); risk-coverage; AURC, lower is better.
- Re-look on a right answer priced from a real re-ask of the right answers
  (on Qwen a crop breaks 133 of 554 right reads), never assumed harmless.
- Re-look rule cross-fitted by figure fold (why: validation split has about 7
  Qwen misreads).

## E5, paragraph 2: results (Table e5, Figure e5)

- Qwen AURC: binary 0.179, controller 0.162, re-look + binary 0.114, oracle 0.056.
- LLaVA AURC: binary 0.645, controller 0.715, re-look + binary 0.638, oracle 0.497.
- Re-look gain, same abstention order: Qwen −0.065 (binary order),
  −0.045 (missing-bar order); LLaVA −0.007, −0.010. All intervals exclude 0.
- Decision-7 controller beats binary for Qwen (−0.017), loses for LLaVA (+0.069).

## E5, paragraph 3: what the re-look actually fixes (do not overclaim)

- Qwen: 212 of 222 fixes are "0" → refusal via verify; only 3 misreads fixed.
  The rule chose verify in every fold, never crop.
- LLaVA: crop fixes 12 of 415 misreads; 437 missing-bar questions re-looked,
  none fixed.
- Abstention order matters more than the re-look: binary order beats
  missing-bar order (it also catches misreads).
- Oracle gap large: knowing the true type is worth far more than the probes give.
- Synthetic only: half the questions ask about a missing bar; ChartQA has almost
  none, so the AURCs do not transfer.
- Source: `results/e5.md`, per-class table and notes.

## Discussion

- Hypothesis verdict needs E3 + E4 + E5 together:
  - E3: type-specific in Qwen, not LLaVA (synthetic); misread vs computation
    separable on real charts, both models (`results/chartqa_type_probes.md`).
  - E4: repair depends on type (pre-registered, both models), with the crop =
    upsample caveat.
  - E5: re-look pays, mostly through verify-induced refusals, not through
    cropping misreads; abstention order dominates.
- Pilot 3 (out-of-range years): add once Yash's probe results are in.
- Limitations for the separate section: LLM annotator (human κ pending),
  synthetic-only E4/E5, small Qwen misread count.
