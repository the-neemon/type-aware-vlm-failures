# cross.md: the cross-intervention matrix (E4, TASKS P6.1 to P6.7)

**Audience:** whoever implements the four interventions and the runner that
produces the recovery matrix. You do not need this project's history to follow
it.

---

## 1. The first thing to know: you are not blocked on anyone

**E4 does not use the probes, and it does not read activations.** That is not an
accident of scheduling, it is the design:

> "This experiment **does not use the probes at all.**" (SPEC Section 4.3)
>
> "E4 is independent of E1 to E3, which is what makes the project robust."
> (TASKS Section 6.1)

The reason is a specific failure mode the project is insuring against. A probe
can be highly accurate at predicting failure type and the distinction can still
buy nothing, because knowing what went wrong does not imply a repair exists.
E4 asks the second question directly, from ground-truth labels, so it stands
whether or not the probes work. If you made E4 depend on probe output you would
destroy the one part of the design that survives a negative E1.

So: **do not wait for the probe code, do not import from `src/probes/`, and do
not read the activation cache.** What you need is the predictions file, the
label file, and GPU time.

What you *do* share with everyone else is the **join key** in Section 3. That is
the only coupling, and it is not optional.

---

## 2. What already exists, so you do not rebuild it

Roughly half of E4 is written. Read these before starting:

| File | What it gives you | Status |
| --- | --- | --- |
| `src/intervene/matrix.py` | the recovery matrix, the pre-registered test, cluster-bootstrapped CIs, argmax stability | **done** |
| `results/e4-preregistration.md` | the test, stated before any data exists, plus a power simulation | **done** |
| `src/intervene/test_matrix.py` | its tests | **done** |
| `docs/taxonomy.md` | what `structural` and `fabrication` mean | done |

**The analysis half is finished. Your job is the four interventions and the
runner that feeds it.** Concretely: P6.1 to P6.5. P6.6 and P6.7 are already
implemented in `matrix.py`.

The interface between your code and that analysis is one dataclass:

```python
@dataclass(frozen=True)
class Outcome:
    """One (failure type, intervention) trial on one erroneous answer."""
    figure_id: str
    failure_type: str      # "structural" or "fabrication"
    intervention: str      # "I_0" | "I_crop" | "I_verify" | "I_abstain"
    recovered: bool
```

Produce a `list[Outcome]` and everything downstream already works:

```python
from src.intervene.matrix import recovery_matrix, cell_counts, did_with_ci

recovery_matrix(outcomes)          # {(type, intervention): P(recovered)}
cell_counts(outcomes)              # {(type, intervention): n}
did_with_ci(outcomes, "I_crop")    # the pre-registered statistic + 95% CI
```

`figure_id` is on `Outcome` because every interval is bootstrapped **by figure,
not by item**. A figure contributes several erroneous answers and their outcomes
are correlated, so item-level resampling gives intervals that are too narrow.
Populate it correctly or every CI in E4 is wrong in the direction that makes
results look significant.

---

## 3. Your inputs, and the one hard contract

### 3.1 The join key

Every artefact in this project joins on `item_id`, derived from the data and
never from a row's position in a file. Position breaks the moment anything is
filtered, sorted or resumed, and it breaks **silently**: the code still runs and
still produces a number.

```python
from src.label.pool import item_id      # do not write your own

iid = item_id("qwen2_5_vl_7b", figure_id, question)
```

Use the literal model string `"qwen2_5_vl_7b"`, the raw unmodified question, and
`figure_id` with no directory. Import the function rather than copying it: two
near-identical definitions of this hash already collided once in this repo and
produced ids that did not join.

### 3.2 Files you read

```
results/predictions/qwen2_5_vl_7b_test.jsonl   # from the inference run, see inf.md
annotations/*.jsonl                            # one file per rater: item_id + label
$HOME/data/ChartQA/test/png/<figure_id>        # the images
$HOME/data/ChartQA/test/tables/<figure_id>.csv # the underlying data table
```

E4's population is **incorrect answers carrying a failure-type label**. So:
take rows with `correct == false`, join the label by `item_id`, and drop
anything labelled `ambiguous`. That mirrors what `src/label/pool.py::build_pool`
already does, and it is worth reading that function rather than re-deriving the
filter.

### 3.3 Match the inference configuration exactly

Every intervention is a **re-query of the same model**. If your re-query differs
from the original run in dtype, attention implementation, image preprocessing or
decoding, the comparison is confounded: you would be measuring a configuration
change, not a repair.

Copy the settings from `inf.md` Section 4, or better, import the same helper the
inference job uses. The ones that silently change behaviour on this hardware:

- `dtype=torch.float16`, **not** bf16. The 2080 Ti is Turing and emulates bf16
  rather than refusing it, so nothing errors and everything is subtly different.
- `attn_implementation="sdpa"`. FlashAttention-2 needs sm_80.
- `use_fast=True` on the processor, `max_pixels=1_000_000`.
- `do_sample=False`, seed 42.
- `--gres=gpu:2`. 16.6 GiB of fp16 weights does not fit an 11 GiB card.

---

## 4. The four interventions

### `I_0`: the baseline. **Read this before writing any code.**

Open decision 5, and it is a trap. Under greedy decoding, re-running the same
prompt reproduces the same wrong answer, so a naive `I_0` recovery rate is
**zero by construction**. Every other intervention then looks good for free, and
the matrix measures nothing.

Two defensible definitions:

1. **`I_0` = keep the original answer.** Recovery is 0 by definition, and you
   state plainly in the paper that it is a definitional floor rather than a
   measurement.
2. **Add a naive re-ask control**: same prompt, resampled at a stated
   temperature. This separates "this repair works" from "asking twice works",
   which is the question a reviewer will ask.

**Recommendation: implement both.** Option 1 is the floor the matrix is written
against; option 2 is the control that makes the other cells interpretable.
Adding a fifth column is cheap; discovering in October that you cannot tell a
repair from resampling noise is not.

This must be decided and written into the paper before P6 runs. Raise it rather
than picking silently.

### `I_crop`: re-query at higher resolution on a question-conditioned crop

Open decision 6. One constraint is **not** open:

> The crop must be conditioned on the question, **never** on the model's own
> attention. Cropping to where a model attended when it misread would reproduce
> the error.

How the crop is derived is open, in increasing order of cost:

1. **Uniform higher-resolution re-render of the whole figure.** Weakest, and the
   honest floor. **Run this regardless**: without it, a gain from option 2
   cannot be attributed to localisation rather than to pixels.
2. **Text matching.** Match question tokens against text in the figure, crop to
   the matched region plus a margin. You do **not** need OCR for this: ChartQA
   ships `test/tables/<figure_id>.csv`, the exact values and category names the
   chart was rendered from. Match against those and map back to the plotted
   region. Cheaper and more accurate than anything recovered from pixels.
3. Gold evidence-region annotations, where the dataset provides them.

Option 1 plus option 2 is the sensible scope. Report them as separate columns
rather than averaging.

Raising resolution means raising `max_pixels` for this intervention only. State
the value you used; it is the manipulated variable.

### `I_verify`: re-query demanding the model name its evidence

Same image, same question, a prompt that requires the model to state which
region of the figure supports its answer before answering. Keep the answer
extraction identical to the main run so `is_correct` still applies.

### `I_abstain`: withheld answer

Scored as **risk avoided, not accuracy gained**. It produces no answer, so it
has no recovery rate, and it is deliberately excluded from the difference in
differences. `matrix.py` raises rather than silently returning a number for it.
Do not add it back. Its value is computed in E5's risk-coverage analysis, which
is a different set of units.

---

## 5. Scoring

Use the existing metric. ChartQA is scored with **relaxed accuracy**, 5 percent
tolerance on numeric answers and exact match on strings:

```python
from src.eval.relaxed_accuracy import is_correct
recovered = is_correct(gold, new_prediction)
```

Do not use exact match, and do not normalise or clean the model's output beyond
stripping whitespace. Both mistakes move items between cells.

---

## 6. What the numbers have to clear

The test is already pre-registered in `results/e4-preregistration.md`, which
matters: it was written before any data existed, so the analysis cannot be
chosen after seeing the matrix. Read it. The short version:

```
delta_crop = [P(ok | structural,  I_crop) - P(ok | structural,  I_0)]
           - [P(ok | fabrication, I_crop) - P(ok | fabrication, I_0)]
```

Prediction: `delta_crop > 0`. Cropping should rescue structural misreadings,
where the answer was on the figure all along, more than it rescues fabrications,
where it never was.

**Power is the binding constraint, so read this before choosing a sample size.**
At 400 items per cell the minimum detectable effect is about **0.12**, which is
large. A true 5 point effect is missed 73 percent of the time. The recommended
target is **800 errors per failure type per model**, which buys 0.94 power at a
10 point effect.

The naturalistic arm cannot supply that: hand-labelling caps it at roughly 1000
items total. So the synthetic arm carries E4's cell counts. This is fine
precisely because **E4 needs inference but not activations**, so scaling it
costs GPU time and nothing in the activation cache.

Report the naturalistic matrix alongside the synthetic one even where its
intervals are wide. A matrix with diagonal structure only on synthetic data is a
much weaker claim, and the paper has to let a reader see that.

---

## 7. Suggested shape

```
src/intervene/interventions.py   I_0, I_crop, I_verify, I_abstain
                                 each: (item, model, processor) -> new prediction
src/intervene/run_matrix.py      CLI: predictions + labels -> outcomes.jsonl
scripts/intervene.sbatch         SLURM wrapper, copy scripts/smoke.sbatch
```

Write `outcomes.jsonl` incrementally with `item_id`, `figure_id`,
`failure_type`, `intervention`, `prediction`, `recovered`, and make the runner
skip completed `(item_id, intervention)` pairs on restart. This is several
thousand forward passes and it will not survive one walltime slot.

Then the analysis is three calls into `matrix.py`, already written and tested.

---

## 8. Things that will cost you a re-run

- Conditioning the crop on model attention. Explicitly forbidden; it reproduces
  the error being repaired.
- Letting `I_0` be a silent zero without saying so, or without a re-ask control.
- Bootstrapping by item instead of by figure. Handled for you if `Outcome`
  carries a correct `figure_id`, and wrong in the flattering direction if not.
- Re-querying with a different dtype, attention implementation or processor
  setting than the original run.
- Folding `I_abstain` into the recovery matrix.
- Scoring with exact match instead of relaxed accuracy.
