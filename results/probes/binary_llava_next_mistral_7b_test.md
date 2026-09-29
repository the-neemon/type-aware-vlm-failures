# E1 binary probe, LLaVA-NeXT (Mistral-7B), ChartQA test (run 28 September)

Raw output: `binary_llava_next_mistral_7b_test.json` (all validation curves) and
`sanity_llava_next_mistral_7b_test.json`. Produced by `src/probes/run_sweep.py`
on Ada gnode026 (CPU), from Shrish's cache
`/home2/shrish.kadam/activations/llava_next_mistral_7b/test.npz` (run 28 Sep on
gnode057, 2x RTX 2080 Ti, fp16, sdpa; validation passed). Same procedure as the
Qwen run (`binary_qwen2_5_vl_7b_test.md`), so the two are directly comparable.

## Setup

- 2,500 items (1,322 correct, 1,178 incorrect; relaxed accuracy 52.9%, human
  44.6%, augmented 61.1%), the same 1,509 figures as Qwen.
- Same figure-level split, seed 42: test set 517 items, **252 errors** (Qwen: 61).
- 2,212 image tokens per question on average (Qwen: 584); 4,096-wide
  activations, 32 layers (Qwen: 3,584, 28).
- Swept 4 pooled positions x 32 layers x L2 in {1, 10, 100, 1000, 10000};
  (position, L2, layer) chosen on **validation only**; test scored once.

## P4.5 sanity probe: PASS

Target: human-written vs augmented question. Best validation AUROC: query_mean
0.999, query_last 0.999, vision_max 0.949, vision_mean 0.948.

## Result

| | LLaVA-NeXT test AUROC [95% CI] | Qwen2.5-VL test AUROC [95% CI] |
| --- | --- | --- |
| **Probe** | **0.798** [0.760, 0.836] (query_last, layer 24 of 32, L2 1000) | **0.877** [0.832, 0.916] (query_last, layer 21 of 28) |
| Surface-only baseline | 0.608 [0.555, 0.656] | 0.715 [0.654, 0.773] |
| Probe minus baseline | +0.190 | +0.162 |
| Probe, human questions only | 0.850 [0.802, 0.896] (135 errors) | 0.829 [0.755, 0.895] (48 errors) |
| Probe, augmented questions only | 0.736 [0.678, 0.790] (117 errors) | 0.878 [0.761, 0.957] (13 errors) |

Validation AUROC at the selected configuration was 0.843, test 0.798.

## Validation AUROC by layer (each position at its best L2)

| Layer | vision_mean | vision_max | query_last | query_mean |
| --- | --- | --- | --- | --- |
| 0  | 0.657 | 0.626 | 0.705 | 0.728 |
| 8  | 0.657 | 0.627 | 0.761 | 0.766 |
| 12 | 0.663 | 0.628 | 0.790 | 0.792 |
| 16 | 0.663 | 0.623 | 0.825 | 0.803 |
| 20 | 0.663 | 0.623 | 0.838 | 0.816 |
| 24 | 0.661 | 0.638 | **0.843** | 0.809 |
| 28 | 0.665 | 0.642 | 0.839 | 0.806 |
| 31 | 0.657 | 0.638 | 0.827 | 0.781 |

## What the numbers support

1. **The E1 result replicates on a second model.** The probe clearly beats the
   surface baseline (intervals do not overlap: probe lower bound 0.760, baseline
   upper bound 0.656), by a margin similar to Qwen's.
2. **Same shape as Qwen.** The signal is in the query-side state and builds with
   depth (query_last 0.71 at layer 0 to a plateau of 0.83 to 0.84 over layers
   20-28); the vision-side positions stay flat near the surface baseline.
3. **It is not reading question style**: within human questions alone it reaches
   0.850. It is weaker within augmented questions (0.736).
4. **The raw AUROC is lower than Qwen's, but the two are not the same task.**
   LLaVA is wrong on 47% of questions against Qwen's 13%, so the two probes
   separate different sets of errors at different base rates, and neither
   model's "errors" have had the not_an_error items removed yet (35% of Qwen's,
   from Naman's labels). Compare the gap over each model's own baseline rather
   than the raw numbers.
