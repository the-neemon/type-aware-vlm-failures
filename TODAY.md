# Today: Wednesday 2 September

One hour each. Pull before you start, push when you stop, even if unfinished.
Each task sits in its own directory, so you should not hit merge conflicts.

Backlog and reasoning live in [TASKS.md](TASKS.md). Calendar constraints are in
Section 3 there, and they are tighter than they look: **the mid-submission
write-up has to be finished by 20 September**, because 21 to 28 September is
midsems.

---

## Yash — environment and weights (P0.2, P0.3)

The single highest-value hour on the project right now. Inference has to be
finished by 10 September or the travel week has no annotation work to do, and
inference cannot start until this is done.

1. Set `HF_HOME` to a path on node-local `/scratch`, **not** `$HOME`. The two
   checkpoints are 29.5 GiB together and the home quota is 30 GiB total, so they
   do not fit alongside anything else. See the budget in
   `configs/activations.yaml`.
2. Write `requirements.txt` with `transformers` and `qwen-vl-utils` pinned to
   exact versions. Record the versions in the file itself, not just the lockfile.
   Activations differ silently across versions, so this is not bookkeeping.
3. Start the Qwen2.5-VL-7B download and leave it running. It does not need
   supervision, and starting it today rather than tomorrow is worth a day.

**Done when:** `requirements.txt` is committed with exact pins, and the download
is running or complete.

**Note:** `scripts/download_chartqa.ps1` is PowerShell and will not run on Ada.
Somebody needs a bash port before the images can land there (decision 11). Not
today unless you finish early.

---

## Shrish — annotation tool (P2.2)

Build the tool now, while there is nothing to annotate, so that the moment the
error set exists on 10 September the annotation can start without a build step
in the way.

Minimal is correct here. Show the figure, the question, the gold answer and the
model's answer; take one keypress for `structural`, `fabrication` or
`ambiguous`; take a one-line rationale; append a row; move on. No web app.

Two requirements that are not optional:
- The annotator must not see which model produced the answer, and must not see
  anyone else's label. Blind annotation is what makes the kappa meaningful.
- Append to disk after every item, never at the end. A crash three hundred items
  in must cost one item.

**Validate it today against the synthetic set**, which already exists and whose
labels are known by construction. Annotate twenty synthetic items and check your
labels against the ground truth. If you cannot agree with a label the generator
assigned by construction, the rubric in `docs/taxonomy.md` has a problem, and
finding that today is much cheaper than finding it on the 11th.

**Done when:** the tool runs, twenty synthetic items are labelled, and you have
compared your labels to the generator's.

---

## Sanjith — generate the synthetic set at scale (P3.6)

The generators are written. Run them for real.

1. **Pull first.** I changed the default seed from 42 to 16 across
   `src/synth/`, which is a standing convention on this project. Anything you
   generated before that pull is not reproducible from the committed defaults,
   so regenerate rather than keeping it.
2. Generate to the targets in `configs/labelling.yaml`: **at least 400 items per
   failure type**, which is what E4 needs in every cell. The synthetic arm now
   carries that load, because hand-labelling caps how much the naturalistic arm
   can supply.
3. Verify the manifest: every figure has a split assignment, no figure appears
   in two splits, and the per-type counts actually clear 400. Assert this in
   code rather than reading the output.

**Done when:** the manifest exists with verified counts and the split assertion
passes.

---

## Naman — surface baseline (P5.4, E2)

Build the E2 control now, before any activations exist, so it cannot get
squeezed later. It needs no cluster and no cached activations.

E2 is the experiment the whole project's credibility rests on. A probe result
without its surface baseline is not a result, and Sahoo et al. is the cautionary
case: 100 percent probe accuracy that collapsed to chance once format confounds
were residualised out.

Build a classifier over surface features alone:
- is the answer string present in the figure's text
- answer type (numeric, categorical, boolean)
- answer length
- question template
- figure type

Train it on the synthetic set, which has labels by construction and exists
today. Report its AUROC. That number is the bar every activation probe will have
to clear, and knowing it before the probes exist keeps everyone honest about
what counts as a result.

**Done when:** the classifier trains on the synthetic set and reports a
per-class AUROC.
