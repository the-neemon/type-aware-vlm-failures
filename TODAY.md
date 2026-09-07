# Today: Wednesday 2 September

One hour each. Pull before you start, push when you stop, even if unfinished.
Each task sits in its own directory, so you should not hit merge conflicts.

Backlog and reasoning live in [TASKS.md](TASKS.md). Calendar constraints are in
Section 3 there. Midsems are 21 to 24 September, which leaves 25 to 30
September as full working days for the write-up, so the real internal deadline
is **experiments done and drafted by 20 September**.

---

## Sanjith — weights on Ada, then generate the synthetic set (P0.3, P3.6)

Two tasks because the first is mostly waiting. Start the download, then generate
while it runs.

**First, 15 minutes.** Set `HF_HOME` to a path on node-local `/scratch`, **not**
`$HOME`. The two checkpoints are 29.5 GiB together and the home quota is 30 GiB
total, so they do not fit alongside anything else; the full breakdown is in
`configs/activations.yaml`. Then start the Qwen2.5-VL-7B and LLaVA-NeXT-7B
downloads and leave them running unattended.

This is the highest-value item on the project right now. Inference has to finish
by 10 September or the travel week has no annotation work, and inference cannot
start until the weights are down. Yash already mapped the filesystem, so the
findings you need are in the repo; nothing here depends on him.

*If you cannot get onto Ada within fifteen minutes, stop, do the generation
below, and say so in the repo so Yash picks the download up. Do not spend the
hour fighting cluster access.*

**Then, the rest of the hour.** Generate the synthetic set at scale.

1. Seed stays at **42**, the project convention, now written into TASKS.md
   Section 4.5 so it does not get changed again. Anything you already generated
   with it is fine.
2. Generate to the targets in `configs/labelling.yaml`: **at least 400 items per
   failure type**, which is what E4 needs in every cell. The synthetic arm now
   carries that load, because hand-labelling caps how much the naturalistic arm
   can supply.
3. Verify the manifest: every figure has a split assignment, no figure appears
   in two splits, and the per-type counts actually clear 400. Assert this in
   code rather than reading the output.

**Done when:** both downloads are running or complete, and the manifest exists
with verified counts and a passing split assertion.

---

## Yash — pin the environment and write the job template (P0.2, P0.5)

Both draw on the node and filesystem knowledge you already have from P0.3, and
neither blocks on the download Sanjith is starting.

1. `requirements.txt` with `transformers` and `qwen-vl-utils` pinned to exact
   versions. Record the versions in the file itself, not just a lockfile.
   Activations differ silently across library versions, so this is not
   bookkeeping: two of us running "the same" job on different versions produce
   different numbers and neither notices.
2. The SLURM template. Three things it must do, all of which cost a rerun if
   missed: request walltime explicitly, because the default is one hour and no
   real job finishes in it; name **one** GPU type in the constraint, because the
   nodes are mixed and a model whose answers and activations come from different
   cards silently breaks the P4.4 coupling; and exclude `gnode077`.
3. Record the chosen GPU type in `configs/activations.yaml` under `gpu_type`,
   which is currently `TBD`. Once set, it is frozen for the project.

**Done when:** `requirements.txt` is committed with exact pins and a job template
exists that a smoke test could use tomorrow.

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

---

## Still unassigned

~~`scripts/download_chartqa.ps1` is PowerShell and will not run on Ada. It needs
a bash port before the images can land there (decision 11, due 5 September).
Whoever finishes early takes it.~~

**Done 7 September by Yash**, whose own P0.2/P0.5 items were already complete.
`scripts/download_chartqa.sh` is the port, and it has been run: ChartQA is on
Ada at `$HOME/data/ChartQA`, 32,719 questions over 20,882 figures, 1.1 GiB.
`python -m src.eval.chartqa ~/data/ChartQA` loads it and the P5.4 no-shared-
figure assertion passes on the real data.

The images are in `$HOME` rather than `/scratch` on purpose: `/scratch` is
node-local and purged at 7 days, so images there would have to be re-downloaded
on whichever node the scheduler picked. The 875 MB zip is staged on `/scratch`
so it never counts against the 30 GiB home quota, and is deleted after
extraction. The script is idempotent, so a job can call it unconditionally.
