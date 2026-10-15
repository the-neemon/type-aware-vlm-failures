# Pilot 3: missing-bar probes on out-of-range years

Run 15 Oct 2026 as a CPU job (Slurm 22404, 11 min), `src/probes/run_pilot3.py`.
Full reports, each with its command line: `results/probes/pilot3_qwen2_5_vl_7b.json`
and `results/probes/pilot3_llava_next_mistral_7b.json`. Pilot 3 data: 300 charts
(`src/synth/out_of_range.py`, manifest sha256 `ed311841...234e2c`, checked by the
job), predictions in `results/predictions/*_synth_pilot3.jsonl`.

    python -m src.probes.run_pilot3 \
        --pilot pilot1 $ACT/<model>/synth_pilot.npz results/predictions/<model>_synth_pilot.jsonl $SYNTH/absent_pilot_v1/manifest.jsonl \
        --pilot pilot2 $ACT/<model>/synth_pilot2.npz results/predictions/<model>_synth_pilot2.jsonl $SYNTH/absent_pilot_v2/manifest.jsonl \
        --target $ACT/<model>/synth_pilot3.npz results/predictions/<model>_synth_pilot3.jsonl $SYNTH/absent_pilot_v3/manifest.jsonl \
        --out results/probes/pilot3_<model>.json

## Outcomes

Each chart is asked "What was the value in <year>?" once for a shown year and once,
in identical words, for a year one to three years outside the shown range.

| Class | Meaning | Qwen | LLaVA |
| --- | --- | --- | --- |
| C | shown year, read correctly | 273 | 74 |
| S | shown year, misread | 27 | 226 |
| R | out-of-range year, rejected | **102** | 0 |
| Z | out-of-range year, "0" | 76 | 19 |
| F | out-of-range year, fabricated number | 105 | **281** |
| U | out-of-range year, unclear (not used as a class) | 17 | 0 |

For comparison, on missing categories (pilots 1 and 2, value questions) Qwen
rejected 15 of 600 and LLaVA 0.

Qwen's rejections depend on direction: 68 of 150 years after the shown range,
34 of 150 before it. LLaVA answers both directions alike (F 140 after, 141 before).

What the fabricated numbers are (5% relative tolerance, as in scoring):

| | Qwen | LLaVA |
| --- | --- | --- |
| fabricated answers | 105 | 281 |
| equal to a shown bar's value | 92 | 135 |
| equal to the series' least-squares extrapolation | 5 | 9 |

No chance baseline is computed for "equal to a shown bar"; with 4 to 7 bars per
chart and LLaVA rounding to multiples of 10, some matches are coincidence.

## Probes

Both probes are trained on the value questions of pilots 1 and 2 only (1,200
items), on query_last. Layer and L2 are chosen by 5-fold out-of-fold AUROC on
pilots 1 and 2 (folds by chart); pilot 3 is scored once with the chosen probe.

- **missing-vs-shown:** missing name vs shown name, whatever the model answered
  (600 vs 600).
- **fabrication:** made-up number for a missing name vs correct read of a shown one
  (the E3 "fabrication" probe; Qwen 91 vs 554, LLaVA 368 vs 185).

AUROC on pilot 3 [95% bootstrap CI over charts]:

| Contrast | Qwen, missing-vs-shown | Qwen, fabrication | LLaVA, missing-vs-shown | LLaVA, fabrication |
| --- | --- | --- | --- | --- |
| selected layer (pilots 1-2 OOF AUROC) | L10 (1.000) | L10 (1.000) | L18 (0.991) | L19 (0.990) |
| out-of-range vs shown (300 vs 300) | **1.000** [1.00, 1.00] | **1.000** [1.00, 1.00] | **0.911** [0.89, 0.93] | **0.880** [0.85, 0.90] |
| R vs C | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | (no R) | (no R) |
| Z vs C | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.994 [0.98, 1.00] | 0.985 [0.96, 1.00] |
| F vs C | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.922 [0.89, 0.95] | 0.901 [0.87, 0.93] |
| F vs Z | 0.477 [0.39, 0.56] | 0.489 [0.40, 0.58] | 0.263 [0.17, 0.37] | 0.290 [0.19, 0.41] |
| F vs R | 0.521 [0.44, 0.60] | 0.528 [0.45, 0.61] | (no R) | (no R) |
| S vs C (both shown) | 0.494 [0.37, 0.62] | 0.499 [0.37, 0.62] | 0.568 [0.49, 0.64] | 0.580 [0.51, 0.65] |

Control: in the first layers every probe is at chance on out-of-range vs shown
(Qwen layers 0 to 2: 0.49 to 0.52; LLaVA layers 0 to 7: 0.44 to 0.54;
`pilot3_absent_vs_shown_by_layer_descriptive_only`), so the year token alone
does not give the answer away. Qwen reaches 1.00 at layer 4 and stays there;
LLaVA rises from layer 8 and holds 0.86 to 0.95 from layer 16.

## What the numbers say

- Both probes, trained only on missing *categories*, fire on out-of-range
  *years*: perfectly for Qwen, at 0.88 to 0.92 for LLaVA. The signal is about
  the question asking for something the chart does not show, not about a name
  being absent.
- In Qwen the probe fires equally whether the model then refuses, says "0" or
  invents a number (F vs R 0.52, F vs Z 0.48): it does not predict which of
  these the model will do. In LLaVA, "0" answers score higher than invented
  numbers (F vs Z 0.26), as on pilots 1 and 2.
- Neither probe responds to misreads of shown years (S vs C 0.49 to 0.58).
- With out-of-range years, fabrication no longer coincides with a missing
  category (AGENTS.md decision 4). Qwen refuses here (102 of 300) where it almost
  never refused missing categories, and most made-up numbers in both models
  equal a shown bar's value rather than an extrapolation of the trend.
