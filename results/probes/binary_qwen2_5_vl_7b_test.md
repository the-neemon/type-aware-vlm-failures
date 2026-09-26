# E1 binary probe, Qwen2.5-VL-7B, ChartQA test (run 26 September)

Raw output: `binary_qwen2_5_vl_7b_test.json` (all validation curves) and
`sanity_qwen2_5_vl_7b_test.json`. Produced by `src/probes/run_sweep.py` on
Ada gnode077, from the cache in `~/anlp/project/vlm-failures-durable/`.

## Setup

- 2,500 items (2,183 correct, 317 incorrect), 1,509 figures.
- Figure-level split, seed 42: train / val / test = 60 / 20 / 20 by figure.
  Test set: 517 items, 61 errors.
- Features z-scored per layer on train statistics only.
- Swept 4 pooled positions x 28 layers x L2 in {1, 10, 100, 1000, 10000};
  (position, L2, layer) chosen on **validation only**; test scored once.

## P4.5 sanity probe: PASS

Target: human-written vs augmented question, which is in the prompt tokens.
Best validation AUROC: query_last 1.000, query_mean 1.000, vision_max 0.958,
vision_mean 0.950. The cache decodes what it obviously contains.

## Result

| | Test AUROC | 95% CI (figure-clustered) |
| --- | --- | --- |
| **Probe** (query_last, layer 21, L2 1000) | **0.877** | [0.832, 0.916] |
| Surface-only baseline | 0.715 | [0.654, 0.773] |
| Probe, human questions only (48 errors) | 0.829 | [0.755, 0.895] |
| Probe, augmented questions only (13 errors) | 0.878 | [0.761, 0.957] |

Validation AUROC at the selected configuration was 0.874, test 0.877, so there
is no visible optimism from selecting among 560 configurations.

## Validation AUROC by layer (each position at its best L2)

| Layer | vision_mean | vision_max | query_last | query_mean |
| --- | --- | --- | --- | --- |
| 0  | 0.654 | 0.651 | 0.687 | 0.691 |
| 6  | 0.654 | 0.643 | 0.716 | 0.703 |
| 12 | 0.651 | 0.674 | 0.761 | 0.733 |
| 15 | 0.653 | 0.666 | 0.810 | 0.777 |
| 18 | 0.657 | 0.675 | 0.851 | 0.819 |
| 21 | 0.659 | 0.682 | **0.874** | 0.811 |
| 24 | 0.663 | 0.668 | 0.868 | 0.773 |
| 27 | 0.665 | 0.681 | 0.848 | 0.756 |

## What the numbers support

1. **The probe beats the surface baseline, and the intervals do not overlap**
   (probe lower bound 0.832, baseline upper bound 0.773).
2. **It is not reading question style.** 80% of errors are human-written
   questions, and question style alone gives 0.679. If the probe only detected
   style it would fall to about 0.5 within each source. It stays at 0.829 among
   human questions and 0.878 among augmented ones.
3. **The signal builds with depth and sits in the query-side state.** query_last
   rises smoothly from 0.69 at layer 0 to a plateau of 0.85 to 0.87 across
   layers 18 to 27. The vision-side positions stay flat at 0.65 to 0.68 at every
   depth, close to the surface baseline. Many neighbouring layers score near the
   peak, so layer 21 is not a lucky single pick.

## What they do not support yet

- **This is not E2.** The surface baseline here uses four easy features (source,
  question length, vision-token count, numeric gold). E2 also requires "answer
  present in figure text", answer type, question template and figure type, plus
  the residualised probe. Until that runs, the claim is "beats a partial
  surface baseline", not "survives E2".
- **One model.** P5.1 asks for both; LLaVA-NeXT has not been run.
- **Augmented has 13 test errors**, hence the wide interval there.
- **Possible tension with the literature.** SPEC 4.1 cites HALP as reporting
  that Qwen2.5-VL-7B is best served by visual-only features. Here the
  query-side state wins clearly. HALP studied hallucination on natural images,
  not chart QA correctness, so these need not conflict, but it should be
  discussed rather than ignored.
