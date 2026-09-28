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
fabrications exist to test transfer (1 of Qwen's 317 ChartQA errors, from
Naman's labels of 28 Sep).

---

# LLaVA-NeXT (Mistral-7B): the same analysis (run 29 Sep)

Same code, charts, folds and settings; caches from Shrish's LLaVA runs of
synthetic pilots 1 and 2 (both passed validation). Reports:
`results/probes/synth_pilot2_llava_next_mistral_7b.json` and
`results/probes/synth_types_llava_next_mistral_7b.json`. LLaVA has 32 layers.

## What LLaVA answered

- **It never rejected a missing name: 0 of 1,200 absent questions** (Qwen: 15).
- Value questions about a missing name (pilot 2): a made-up number 217, "0" 83.
  Lookalikes raise the made-up rate (120/143 vs 97/157), but the numbers are
  mostly round values near some bar (10, 40, 160), not copies of the lookalike's
  value as with Qwen.
- Shown bars: value 100/300 right, compare 209/300, neighbor 255/300. LLaVA snaps
  readings to the nearest gridline (52 -> 50, 34 -> 30), so most value reads miss
  relaxed accuracy's 5% tolerance.

Value-question classes, both pilots: **C 185, S 415, F 368, Z 232** (Qwen: 554,
46, 91, 494).

## Absent vs shown name (single split, as for Qwen)

| Family | LLaVA test AUROC [CI] | Where | Qwen |
| --- | --- | --- | --- |
| value | 0.977 [0.94, 1.00] | query_mean L12 | 0.999 |
| compare | 0.959 [0.92, 0.99] | query_mean L19 | 0.991 |
| neighbor | 0.996 [0.99, 1.00] | query_last L18 | 1.000 |

Text baseline 0.49 to 0.52 and vision positions 0.500, as for Qwen. Lookalike
names only: 0.951, 0.922, 0.991. The signal appears later than in Qwen
(validation about 0.5 up to layer 3, 0.96 by layer 9; Qwen: 0.99 from layer 2).

## The type matrix (5-fold CV by chart)

| Probe (trained on) | S vs C | F vs C | Z vs C | F vs Z |
| --- | --- | --- | --- | --- |
| structural (S vs C) | **0.776** [0.73, 0.82] | 0.879 [0.85, 0.90] | 0.987 [0.98, 0.99] | 0.215 [0.18, 0.25] |
|   metadata baseline | 0.512 | 0.522 | 0.486 | 0.537 |
| fabrication (F vs C) | 0.680 [0.63, 0.73] | **0.988** [0.98, 0.99] | 1.000 | 0.243 [0.21, 0.28] |
|   metadata baseline | 0.642 | 0.769 | 0.643 | 0.651 |
| fab_vs_zero (F vs Z) | 0.520 [0.47, 0.57] | 0.226 | 0.000 | **0.986** [0.98, 0.99] |
|   metadata baseline | 0.566 | 0.637 | 0.533 | 0.600 |

Layers chosen: structural 20-30, fabrication 16-19, fab_vs_zero 23-30.

## LLaVA against Qwen

1. **The missing-bar signal replicates.** Both models represent that the asked
   name is not on the chart (0.96 to 1.00) and answer anyway; LLaVA never
   refuses. In both, the fabrication probe is this missing-bar detector (Z vs C
   1.000) and is the more type-specific of the two: on misreads it scores 0.576
   in Qwen and 0.680 in LLaVA.
2. **The misread signal does not separate in LLaVA.** Its structural probe fires
   more on missing-bar questions (F vs C 0.879, Z vs C 0.987) than on its own task
   (0.776). In LLaVA it behaves like a general "this answer is not grounded"
   direction, strongest when there is no bar at all. In Qwen it was specific
   (0.887 own, 0.594 on fabrications). By SPEC E3's criterion, LLaVA shows
   strong cross-transfer from the structural side: one signal wearing two labels,
   in one direction only.
3. **LLaVA's misread probe does not beat a one-number chart baseline.** Because
   LLaVA snaps to gridlines, whether a read is "correct" depends largely on how
   close the true value is to a multiple of 10. That single number predicts
   LLaVA's misreads at AUROC 0.852 (all 600 shown-bar value questions, no fitting),
   above the probe's 0.776. For Qwen the same number gives 0.616, well below its
   probe's 0.887. The metadata baseline in the table does not include it, which is
   why it looks weak for LLaVA (0.512). Caveat: the feature was chosen after seeing
   LLaVA's rounding, so it is a diagnostic, not a pre-registered baseline.
4. **Made-up number vs "0" is decodable in both** (0.986 LLaVA, 0.984 Qwen), late
   in both (layers 23-30 and 22-27): mostly a readout of the imminent answer.

## Limits (in addition to those above)

LLaVA's correct class is small (185) and made of values that happen to sit near
gridlines. The structural comparison between the two models therefore rests on
very different class balances (C:S = 185:415 vs 554:46).
