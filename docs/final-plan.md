# Final submission plan (14 to 31 Oct 2026)

Read `AGENTS.md` first: its decisions and rules apply to every task here. Work
through your person's section in order. A task is done when its completion
criterion holds, and only then. When a task waits on someone else's output,
work on your next task meanwhile.

**Critical path:** E4 runs (Shrish, by 18 Oct) → E4 analysis (Naman, 20 Oct) →
E5 (Naman, 24 Oct) → writing (25 to 28 Oct) → merge and submit (29 to 30 Oct).
Pilot 3 has its own chain: generator (Naman, 17 Oct) → runs (Shrish, 22 Oct) →
probes (Yash, 24 Oct). A slip on this path moves the submission, so say so in the team
chat the day it happens.

## What the final submission contains

`HiddenStates-Final.zip` with: the report (at most 8 pages excluding references,
plus up to 2 appendix pages; limitations and future work in their own section
just before the references, holding no results), the code as a ZIP archive, and
the slides as a PDF. HuggingFace or WandB links go in the README only if we
upload something; currently we do not.

## Shrish: GPU runs on Ada, annotator A

1. **Done 15 Oct: nothing uncommitted.** `$DURABLE/e4/` holds only the four
   E4 outcome files, identical to `results/e4/` (same line and byte counts).
   Kept for the record: **Check for existing output (14 Oct).** Look on Ada for any E4 output
   (`$DURABLE/e4/`) or other results not yet in the repo. Done when every
   finished run is either committed or known not to exist; tell the team which.
2. **Done 14 Oct: the caches were already readable** from Naman's account (group
   `research`), so nothing to do. Kept for the record: **Share the caches with
   Naman (14 Oct).** Naman's agent runs the LLaVA
   residualised probes and E5 as CPU jobs under Naman's Ada account, reading
   the existing activation caches. Done when Naman's account can read the
   ChartQA and synthetic `.npz` caches of both models (group-readable, or a
   copy in Naman's `$HOME`) and Naman has their paths.
3. **Done 14 Oct: E4 had already run on 1 Oct** (jobs 861 to 864); the outcomes
   are in `results/e4/` and the analysis in `results/e4.md`. Kept for the record:
   **Run E4 for both models (14 to 18 Oct).** Submit
   `scripts/intervene.sbatch` in synthetic mode (header shows the variables)
   for each model's pilot 1 and pilot 2 predictions. First update its
   `--exclude` line to the list in `AGENTS.md`. Jobs resume after a timeout, so
   resubmit the same command until each finishes. Done when the outcomes files
   cover all 631 Qwen and 1,015 LLaVA value-question errors under every
   intervention, and are committed under `results/e4/` (small JSONL files; no
   activations).
4. **Done 15 Oct: all four `*.correct.outcomes.jsonl` files are in
   `results/e4/`** (Qwen 295 + 274, LLaVA 85 + 100 items, every repair;
   provenance in `results/e4.md`). Kept for the record:
   **Re-ask the correctly answered value questions for E5 (15 to 17 Oct).**
   E4 re-asked only wrong answers; E5 also needs to know whether a repair keeps
   a right answer right. `git pull` on Ada first, then for each model and each
   of pilots 1 and 2:

       PREDICTIONS=results/predictions/<model>_synth_pilot.jsonl \
       SYNTH_MANIFEST=$HOME/data/synth/absent_pilot_v1/manifest.jsonl \
       CORRECT=1 sbatch scripts/intervene.sbatch
       # pilot 2: <model>_synth_pilot2.jsonl and absent_pilot_v2

   About 570 Qwen and 185 LLaVA questions, four repairs each; resubmit the same
   command after a timeout. Done when the four `*.correct.outcomes.jsonl` files
   from `$DURABLE/e4/` are committed under `results/e4/` and Naman knows.
5. **Label the 200 agreement items (15 to 19 Oct)** as annotator `shrish` in
   `streamlit run src/label/app.py`, once Sanjith's task files exist, while the
   E4 jobs run. A person does this; an agent only starts the app. Done when all
   200 are labelled.
6. **Done 15 Oct: pilot 3 predictions are in `results/predictions/`**
   (`*_synth_pilot3.*`, 600 items each) and both caches exist on Ada as
   `~/activations/<model>/synth_pilot3.npz` (jobs 22164 Qwen, 22165 LLaVA;
   manifest hash in `configs/inference_synth_pilot3.yaml`). LLaVA's run failed
   only its last step, the ChartQA-based accuracy floor (answerable 0.247
   against 0.27); its summary keeps `"passed": false`. Its present-bar answers
   are all numbers, and 286 of 300 are multiples of 10, the same gridline
   reading as pilots 1 (85/300 correct) and 2 (100/300). Kept for the record:
   **Run pilot 3 (18 to 22 Oct)**, once Naman's generator lands: inference and
   activation caching for both models, the same way as pilot 2. Done when both
   predictions files are in `results/predictions/` and both caches exist on
   Ada, with the cache paths sent to Yash.
7. **Done 15 Oct: no shard folders left on Ada**; every `.npz` kept. Only the
   pilot 3 shards (both models, 1.06 GB) remained; the pilot 1 and 2 ones were
   already gone. Kept for the record:
   **Delete the leftover LLaVA shard folders on Ada**
   (`~/activations/*/shards_synth_pilot*`), keeping the final `.npz` files.
8. **Write (25 to 28 Oct):** experimental setup, and the reproducibility
   appendix.

## Yash: pilot 3 probes, report lead

1. **Done 15 Oct: `results/pilot3_probes.md`** (reports
   `results/probes/pilot3_<model>.json`, code `src/probes/run_pilot3.py`). Both
   missing-bar probes from pilots 1 and 2 fire on out-of-range years: Qwen 1.000,
   LLaVA 0.911 (missing-vs-shown) and 0.880 (fabrication), with CIs. Kept for the
   record: **Pilot 3 probes (22 to 24 Oct)**, once Shrish's caches exist: the outcome
   table (rejected, "0", fabricated) and whether the missing-bar probe from
   pilots 1 and 2 fires on out-of-range questions. Done when a result file in
   `results/` answers that with AUROC and confidence intervals for both models.
2. **Write (14 to 29 Oct):** start now on the updated introduction and related
   work (title change per decision 1), then the synthetic arm and E3. Merge everyone's sections into `paper/final/` and bring the
   report to the page limit. Done when it compiles without errors at 8 pages or
   fewer before the references.
3. **Upload (30 Oct)** the zip Naman builds.

## Naman: agent-run code and analysis


1. **Pilot 3 generator (14 to 17 Oct)**, per decision 4 in `AGENTS.md`. Build it
   beside `src/synth/absent_pairs.py`, reusing its drawing, validation and
   `--verify` pattern, with tests like `test_absent_pairs.py`. Done when 300
   charts generate deterministically with seed 42, `--verify` passes, the tests
   pass, and Shrish has the command.
2. **LLaVA residualised type probes (14 to 16 Oct)**, the same procedure as
   Qwen's in `results/chartqa_type_probes.md`, as a CPU job on Ada. Done when
   the LLaVA column of the residualised rows is filled and committed. This
   closes E2.
3. **E4 analysis (19 to 20 Oct)**, once Shrish's outcomes are committed: the
   recovery matrix by failure type, and the pre-registered test with
   `src/intervene/matrix.py`. Done when `results/e4.md` reports the full matrix
   for both models and the `I_crop` difference in differences with its 95%
   cluster bootstrap interval, stated against the pre-registered prediction
   whether it holds or fails.
4. **E5 controller (21 to 24 Oct)**, per decision 7 in `AGENTS.md`: abstain when
   the missing-bar probe fires, re-look with the best E4 repair when the binary
   or misread probe fires. Done when `results/e5.md` has risk-coverage curves
   and AURC for the controller, binary-only abstention and the oracle, both
   models.
5. **Code release (23 to 27 Oct).**
   - Replace every `/home2/<user>` path, node name and cluster hostname in
     tracked files with a variable (`git grep -n -E "/home2|gnode[0-9]|ada\.iiit"`
     lists them; 39 files at the mid submission). Done when that grep returns
     nothing and the tests still pass.
   - README: setup, and the command that reproduces each result file. Done when
     every result file the report cites has its command in the README.
6. **Report tables and figures (25 to 28 Oct):** LaTeX tables and figures for
   E4, E5 and pilot 3 from the result files, handed to the section writers. No
   prose.
7. **Write (25 to 28 Oct):** E4, E5 and the discussion.
8. **Number check (29 Oct):** every number in the merged draft matched against
   `results/`. Done when each mismatch is listed for its section's writer.
9. **Package (30 Oct):** build `HiddenStates-Final.zip` from the final report,
   the code zip (`git archive` of `main`) and Sanjith's slides, and verify it.

## Sanjith: labelling lead, annotator B, slides

1. **Set up the agreement check (14 to 15 Oct)**, per decision 6 in
   `AGENTS.md`. Draw the 200 items with seed 42, write them as a pool, then
   `python -m src.label.pool assign --pool <your pool.jsonl> --annotators shrish sanjith
   --overlap 200 --out annotations/tasks`.
   Done when both task files exist and neither shows the model or the LLM label.
2. **Label the 200 items (15 to 19 Oct)** as annotator `sanjith`. A person does
   this; an agent only starts the app. Done when all 200 are labelled.
3. **Agreement result (20 to 21 Oct).** Done when `results/human_agreement.md`
   reports κ per model and overall, the human-vs-LLM agreement, and the
   adjudicated disagreements.
4. **Write and assemble slides (25 to 30 Oct):** the labelling and data
   section, and the limitations section. Collect each person's slides for their
   part into one deck in `paper/final/`. Done when the deck builds to PDF.

## Shared final phase

- **25 to 28 Oct:** each person writes their own sections (see `AGENTS.md` on
  prose); Naman's agent supplies tables, figures and numbers.
- **29 Oct:** Yash merges; Naman's agent checks every number against
  `results/`; the writers fix their sections.
- **30 Oct:** submit. **31 Oct:** buffer and viva preparation.

## Only if the critical path finishes early

More Qwen misreads (about 1,000 charts on 0 to 200 axes with sparse ticks); the
"0" diagnostic (re-ask the absent questions without the answer suffix); a
causal steering test along the missing-bar direction.
