# Type-aware probes on synthetic value questions (E3, synthetic arm)

Run 28 Sep on gnode028 (CPU only), `src/probes/run_synth_types.py`; full report
`results/probes/synth_types_qwen2_5_vl_7b.json`.

## Data

Every synthetic "What is the value of X?" item from pilots 1 and 2 (the one
template where both failure types occur under identical wording), scored with
the current scorer:

| Class | Meaning | n |
| --- | --- | --- |
| C | correct read of a bar on the chart | 554 |
| S | structural: misread of a bar on the chart | 46 |
| F | fabrication: a made-up number for a name not on the chart | 91 |
| Z | "0" for a name not on the chart (refusal expressed as a number, not fabrication) | 494 |

15 rejections ("None") are left out.

## Method

Linear probes on the query_last activation. 5-fold cross-validation by figure:
every item is scored by a probe that never saw its chart. Within each fold, layer
(0-27) and L2 are chosen by inner cross-validation on the other folds. Baseline:
the same probes on chart metadata only (pilot, axis range, tick density, bar
count, lookalike name present, phrasing). 95% CIs by figure bootstrap.

## Result: each probe applied to each contrast (AUROC)

| Probe (trained on) | S vs C | F vs C | Z vs C | F vs Z |
| --- | --- | --- | --- | --- |
| structural (S vs C) | **0.887** [0.82, 0.94] | 0.594 [0.55, 0.65] | 0.673 [0.64, 0.70] | 0.400 [0.34, 0.46] |
|   metadata baseline | 0.814 [0.76, 0.86] | 0.557 | 0.515 | 0.543 |
| fabrication (F vs C) | 0.576 [0.50, 0.66] | **1.000** [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.286 [0.22, 0.35] |
|   metadata baseline | 0.588 | 0.840 [0.78, 0.90] | 0.564 | 0.787 |
| fab_vs_zero (F vs Z) | 0.336 [0.26, 0.41] | 0.138 | 0.000 | **0.984** [0.97, 0.99] |
|   metadata baseline | 0.582 | 0.831 | 0.559 | 0.783 [0.72, 0.84] |

Layers chosen: structural 16-25, fabrication 9-13 (own-task AUROC is already
0.97 at layer 3), fab_vs_zero 22-27.

## What it shows

1. **The two failure signals are largely separate.** Each probe is strong on its
   own failure type and weak on the other: structural 0.887 on misreads but 0.594
   on fabrications; fabrication 1.000 on fabrications but 0.576 (CI touching 0.5)
   on misreads. That is the diagonal pattern the type-aware controller needs.
2. **The fabrication probe detects that the asked-about bar is missing, not the
   invented number itself.** It scores "0" answers exactly as high as made-up
   numbers (Z vs C 1.000), and ranks "0" above them (F vs Z 0.286). It works from
   layer 3 on. For a controller that is the useful signal: it flags every question
   Qwen cannot answer from the chart, although Qwen answers almost all of them.
3. **Which wrong answer Qwen gives to a missing bar (a made-up number or "0") is
   decodable, but late** (0.984 vs 0.783 from metadata, best at layers 22-27). It
   mostly reads out the answer about to be produced. Its direction orders items C
   > F > Z and scores misreads below correct reads (0.336), so it behaves like
   "the model has a value to report" more than a fabrication-specific signal.
4. **The structural probe beats chart metadata by 0.07** (0.887 vs 0.814, CIs
   overlap). Much of misread risk is chart difficulty (0-200 axes, sparse ticks)
   that metadata already captures. It also scores "0" items somewhat above
   correct ones (0.673), a partial overlap with the missing-bar signal.

## Limits

Synthetic bar charts only; one question template; 46 structural errors; the
fabrication class is defined by a missing name, so "fabrication probe" and
"missing-bar detector" cannot be separated with this data. No naturalistic
fabrications exist to test transfer (0 in the 20 labelled ChartQA errors).
