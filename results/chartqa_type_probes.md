# Type-aware probes on ChartQA with the labelled errors (E3, real-chart arm)

Run 1 Oct 2026 on gnode002 (CPU only, Slurm job 106), `src/probes/run_chartqa_types.py`.
Full reports: `results/probes/chartqa_types_llava_next_mistral_7b.json` and
`results/probes/chartqa_types_qwen2_5_vl_7b.json`. This is the ChartQA counterpart
of `results/synth_type_probes.md`: the same folds, layer selection and probes, but
the classes come from the five-label annotations instead of synthetic outcomes.

## Data

Every ChartQA test item, with each error given its label (Naman for Qwen, Sanjith
for LLaVA). `ambiguous` errors are left out; no error is unlabelled.

| Class | Meaning | LLaVA | Qwen |
| --- | --- | --- | --- |
| C | correct (relaxed accuracy) | 1,322 | 2,183 |
| S | structural: answer explained by misreading the figure | 583 | 71 |
| K | computation: inputs read right, arithmetic or logic wrong | 435 | 93 |
| F | fabrication: content the figure does not support | 17 | 1 |
| N | not_an_error: scored wrong, actually right | 69 | 112 |

## Method

Linear probes (L2 logistic regression) on the query_last activation:

- **structural**: S vs C
- **computation**: K vs C
- **struct_vs_comp**: S vs K. This tests whether the two error types ChartQA
  actually produces are linearly separable.

There is no fabrication probe. A probe needs at least 20 positive items, and there
are 17 fabrications for LLaVA and 1 for Qwen.

Every item is scored out of fold: there are 5 folds by figure, so no probe scores a
chart it was trained on. Within each fold, layer and L2 are chosen by inner
cross-validation on the other folds.

Each probe is then applied to every contrast. A row is a probe and a column is a
pair of classes. AUROC below 0.5 means the probe ranks the second class higher.
For example, the computation probe on S vs K gives 0.04, meaning it rates
computation errors far above misreads, which is what it should do.

Baseline: the same probes trained on four surface features only, with no
activations: question source (human or augmented), question length, number of
image tokens, and whether the gold answer is a number. The 95% CIs come from a
bootstrap over figures.

## Result: LLaVA-NeXT (AUROC, 95% CI)

| Probe | S vs C | K vs C | S vs K | F vs C (17) | N vs C (69) |
| --- | --- | --- | --- | --- | --- |
| structural | **0.813** [0.79, 0.83] | 0.586 [0.55, 0.62] | 0.768 [0.74, 0.80] | 0.752 [0.64, 0.85] | 0.490 [0.41, 0.57] |
| (surface) | 0.536 | 0.389 | 0.655 | 0.548 | 0.331 |
| computation | 0.540 [0.51, 0.57] | **0.949** [0.94, 0.96] | 0.041 [0.03, 0.05] | 0.663 [0.52, 0.80] | 0.650 [0.59, 0.71] |
| (surface) | 0.484 | 0.821 | 0.175 | 0.521 | 0.501 |
| struct_vs_comp | 0.610 [0.58, 0.64] | 0.085 [0.07, 0.10] | **0.961** [0.95, 0.97] | 0.429 [0.29, 0.57] | 0.382 [0.32, 0.44] |
| (surface) | 0.566 | 0.229 | 0.845 | 0.514 | 0.368 |

Layers chosen per fold: structural 21 to 23; computation 19 to 21; struct_vs_comp
14 to 23.

## Result: Qwen2.5-VL (AUROC, 95% CI)

| Probe | S vs C | K vs C | S vs K | N vs C (112) |
| --- | --- | --- | --- | --- |
| structural | **0.892** [0.85, 0.93] | 0.820 [0.79, 0.85] | 0.711 [0.63, 0.79] | 0.679 [0.62, 0.73] |
| (surface) | 0.651 | 0.765 | 0.393 | 0.644 |
| computation | 0.732 [0.66, 0.80] | **0.904** [0.87, 0.93] | 0.221 [0.15, 0.29] | 0.678 [0.62, 0.73] |
| (surface) | 0.666 | 0.818 | 0.284 | 0.665 |
| struct_vs_comp | 0.581 [0.52, 0.64] | 0.168 [0.14, 0.20] | **0.897** [0.84, 0.94] | 0.410 [0.36, 0.47] |
| (surface) | 0.378 | 0.228 | 0.693 | 0.351 |

Layers chosen per fold: structural 20 to 24; computation 18 to 26; struct_vs_comp
15 to 18.

## Also run: LLaVA binary probe without the not_an_error items

This repeats the Qwen check in
`results/probes/binary_without_not_an_error_qwen2_5_vl_7b_test.json`. Dropping
the 69 not_an_error items moves LLaVA's binary probe from 0.798 to **0.819**
[0.782, 0.853] (surface baseline 0.636; L24, as before). Within source, it scores
0.861 on human questions and 0.767 on augmented ones.

For Qwen the same step went from 0.877 to 0.932. The gain is smaller for LLaVA
because not_an_error is 6% of its errors, against 35% of Qwen's. Report:
`results/probes/binary_without_not_an_error_llava_next_mistral_7b_test.json`.

## What it shows

1. **Misreads are detectable on real charts, for both models, well above the
   surface features.** The structural probe scores 0.813 for LLaVA (surface
   0.536) and 0.892 for Qwen (surface 0.651). Qwen's number agrees with the
   earlier single-split result (0.905, 13 test positives), now measured on all
   71 misreads.

2. **Misreads and computation errors are separable.** The struct_vs_comp probe
   scores 0.961 for LLaVA and 0.897 for Qwen, against surface baselines of 0.845
   and 0.693. This is the real-chart version of the synthetic misread vs
   fabrication result. The two error types ChartQA produces in quantity are
   represented differently.

3. **LLaVA's probes are type-specific; Qwen's structural probe is not.** LLaVA's
   structural probe barely responds to computation errors (K vs C 0.586), and its
   computation probe barely responds to misreads (S vs C 0.540): a clean diagonal.
   Qwen's structural probe also flags computation errors (K vs C 0.820). However,
   the surface features alone already give 0.765 there, so much of this is
   question style rather than a shared "error" direction. With 71 and 93
   positives, Qwen's off-diagonal cells are less certain than LLaVA's.

4. **Not_an_error items look like correct ones to LLaVA's structural probe**
   (N vs C 0.490), which is the expected sanity result: those answers were right.
   For Qwen, N vs C is about 0.68 for both type probes, but the surface features
   give 0.644. Nearly half of Qwen's not_an_error items (51 of 112) are formatting
   mismatches, which depend on the question style.

5. **LLaVA's 17 fabrications look more like misreads than like correct answers**
   (structural probe 0.752 [0.64, 0.85]). The interval is wide, and this is
   descriptive only, not a fabrication probe.

## Caveat: much of the computation signal is the question, not the error

The computation probe is strong (0.949 LLaVA, 0.904 Qwen), but the surface
baseline already reaches 0.821 and 0.818. The probe at layer 0, before the model
has done any real processing, gives 0.865 (LLaVA) and 0.769 (Qwen). The cause is
that computation errors only happen on questions that ask for arithmetic
("difference", "sum", "ratio", "average"), and the question's wording is visible
in the activations from the first layer.

So "K vs C" partly measures "is this an arithmetic question", not only "will the
arithmetic go wrong". The same applies to struct_vs_comp, whose layer-0 AUROC is
0.907 (LLaVA) and 0.716 (Qwen). The structural probe is less affected: its
layer-0 AUROC is 0.757 for LLaVA and 0.805 for Qwen, against 0.813 and 0.892 at
the chosen layers.

A fairer contrast for computation would compare computation errors with
*correct answers to arithmetic questions only*. That needs a question-type tag on
every item (a keyword rule would do), and is the natural next step. The same
restriction for misreads (retrieval questions only) would tighten the structural
result too.

## Limits

- One annotator per model (an AI-assisted labelling pass, audited). The two-person
  agreement check in `todos.md` is still owed.
- Qwen's classes are small (71 S, 93 K), so its CIs are wide.
- The surface baseline has only four features. As the layer-0 numbers show, it
  understates how much the question alone gives away.
