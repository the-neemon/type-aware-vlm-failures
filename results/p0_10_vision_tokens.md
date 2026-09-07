# P0.10: vision-token counts verified against the real processor

Run 7 September 2026 on Ada, gnode055, inside the pinned environment
(`transformers==4.57.6`). Command:

```bash
python scripts/verify_vision_tokens.py --max-pixels 1000000 <5 ChartQA figures>
```

**Result: PASS**, exit code 0. All five figures matched exactly.

## What was being checked

`configs/activations.yaml` and the storage budget in `results/storage_budget.md`
rest on vision-token counts that `src/extract/storage_budget.py` derives
analytically, by reimplementing Qwen's `smart_resize` from the config. Nothing
had confirmed that reimplementation against `Qwen2VLImageProcessor` itself. An
off-by-one in the patch merge or a different rounding rule would have shifted
the whole budget table, and the failure would have been quiet.

## Measured

Five real ChartQA test figures, chosen at stride across the split so they differ
in pixel size rather than being five crops of one source.

| Figure | Size | Analytic | Actual | Match |
| --- | --- | --- | --- | --- |
| `chartqa_test_000000.png` | 850x600 | 630 | 630 | yes |
| `chartqa_test_000500.png` | 415x322 | 180 | 180 | yes |
| `chartqa_test_001000.png` | 800x557 | 580 | 580 | yes |
| `chartqa_test_001500.png` | 800x557 | 580 | 580 | yes |
| `chartqa_test_002000.png` | 800x557 | 580 | 580 | yes |

**Observed range: 180 to 630 tokens.** 800x557 is the common ChartQA test size
and gives 580.

The previous planning number in `configs/activations.yaml` was 1044, from a
hypothetical 800x1000 figure. It is pessimistic for real ChartQA and has been
replaced with the measured 580, with the range recorded alongside it.

**This does not move the budget.** Pooled cost per item is
`vectors_per_layer x hidden_size x n_layers x 2 bytes` and does not reference
the token count at all. The count only ever fed the Schema C (unpooled)
estimate, which was rejected as unaffordable regardless. So the correction makes
the rejected option look slightly cheaper than stated and changes nothing about
the decision. `layers: all` stands.

## Secondary finding: the image processor default changed, and it is not free

`transformers` 4.57 loads `Qwen2VLImageProcessorFast` by default and emits:

> The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast
> processor by default ... This is a breaking change and may produce slightly
> different outputs.

It does. Running both processors over the same five figures:

| Figure | Patches (fast) | Patches (slow) | max abs diff | mean abs diff | Identical |
| --- | --- | --- | --- | --- | --- |
| `chartqa_test_000000.png` | 2520 | 2520 | 0.030015 | 0.000009 | no |
| `chartqa_test_000500.png` | 720 | 720 | 0.028440 | 0.000020 | no |
| `chartqa_test_001000.png` | 2320 | 2320 | 0.015008 | 0.000007 | no |
| `chartqa_test_001500.png` | 2320 | 2320 | 0.015008 | 0.000008 | no |
| `chartqa_test_002000.png` | 2320 | 2320 | 0.030016 | 0.000006 | no |

Token counts are identical, so P0.10 itself is unaffected either way. The
**pixels** are not identical: worst absolute difference 0.030 on normalised
values, and no figure came out bit-identical.

The differences are small, and on accuracy they would almost certainly be
invisible. That is exactly why this matters here. This project's claim is that
small differences in internal state carry information about failure type, so
"small enough not to change the answer" is not the relevant bar. Two team
members on different `transformers` defaults would cache different activations
from the same figure, compare them, and neither would see a reason to suspect
the pipeline. It is the same failure mode `requirements.txt` exists to prevent.

`processor_use_fast: true` is now recorded in `configs/activations.yaml`
alongside `gpu_type` and `gpu_dtype`, and is passed explicitly in both
`src/extract/smoke_qwen.py` and `scripts/verify_vision_tokens.py` rather than
left to the library default. `true` is the 4.57 default and the maintained
path, so this pins current behaviour rather than opting into legacy behaviour.

## Note on the job template

`scripts/verify_tokens.sbatch` no longer calls `stage_in_model`. This check
needs only the processor files, a few MB that `AutoProcessor` fetches itself;
staging the full checkpoint downloaded roughly 16 GB to support a
thirty-second check.
