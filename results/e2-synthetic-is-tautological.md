# E2 cannot be run on the synthetic arm

Found while building the surface baseline (P5.4) on 2 September.

## The result

Surface features alone, per-class AUROC on 600 synthetic items with a
figure-level split:

| Feature set | structural | fabrication |
| --- | --- | --- |
| All surface features | 1.000 | 1.000 |
| `question_template` alone | 1.000 | 1.000 |
| `answer_in_figure` alone | 1.000 | 1.000 |
| Neither of those two | 0.953 | 0.953 |

## Why, and why it is not a bug

Two features each separate the classes perfectly, and both do so by
construction rather than by accident.

`question_template` determines the label. In `src/synth/questions.py` the
mapping is fixed: `read_value`, `compare_bars` and `bar_difference` are tagged
`structural`, `absent_category` is tagged `fabrication`. Predicting the label
from the template is reading the generator's source.

`answer_in_figure` separates them for the same underlying reason. A fabrication
item asks about a category that is not on the chart, so no answer to it can
appear in the figure's text. The feature is a restatement of the item's design.

The 0.953 row is the least trustworthy number here, because it reflects the
placeholder answers this was run on rather than real model output. It moved from
0.939 to 0.953 when the placeholder was changed, which is the point: treat it as
noise until inference lands. The two 1.000 rows did not move, and do not depend
on the placeholder at all.

## What follows

**E2's evidence has to come from the naturalistic arm.** The probe-surface gap
is the project's central control, and on synthetic data the surface baseline is
pinned at 1.000, so the gap can only ever be zero or negative there. Reporting
a synthetic E2 would not be a weak result, it would be a meaningless one.

This interacts with decision 2 in a way worth stating plainly. Losing the
labelling budget shrank the naturalistic arm to roughly 1000 hand-labelled
items, and it turns out that is the *only* arm on which E2 can run at all. So
hand-labelling now bounds the statistical power of the control that everything
else is reported against, not just the sample size of one experiment.

Two consequences for how the naturalistic annotation is run:

- The double-annotated set matters more than it did. E2 on noisy labels
  understates the surface baseline and flatters the probe.
- If the 1000-item target has to be cut, cut synthetic generation instead.
  Synthetic items are free and E4 needs them, but they cannot substitute for a
  single naturalistic item where E2 is concerned.

## Caveat

Run on placeholder answers, since inference (P1.3, P1.4) has not happened. The
two structural findings do not depend on the placeholder, but every number here
should be regenerated against real predictions before it is quoted:

    PYTHONPATH=. python3 -m src.surface.run_e2_baseline <synth_dir>
