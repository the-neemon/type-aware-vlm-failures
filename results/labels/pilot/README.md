# Labelling pilot (P2.3): 20 Qwen2.5-VL-7B ChartQA test errors

Labeller: Claude (claude-opus-5-5) in a Claude Code session, 26 September.
Input per item, the same as the human annotation app shows: the chart image, the
question, the gold answer and the model's answer. The ChartQA data tables were
deliberately **not** used, so these labels are comparable with human ones.
Rubric: `docs/taxonomy.md` as of this date. Items: the first 20 of the 317
errors in a seed-42 shuffle (14 human-written, 6 augmented).

Labels with one-line rationales: `claude_pilot20_qwen2_5_vl_7b_test.jsonl`.
These are **pilot** labels and not in `results/labels/` proper; the rubric
should be revised before labelling continues (below).

## Result

| Label | Count |
| --- | --- |
| structural | 6 |
| fabrication | **0** |
| ambiguous | 14 |

The 14 ambiguous items, by reason:

| Reason | Count | Model's answer actually correct? |
| --- | --- | --- |
| scoring artifact (1:2 vs 0.5; 3 vs 0.03; a footnote asterisk) | 3 | all 3 |
| gold answer wrong or unexplainable | 4 | 2 of 4 |
| question admits several readings | 5 | 1 of 5 |
| computation or reasoning on correctly read values | 2 | 0 |

In **6 of the 20** "errors" the model's answer is right and the scoring or the
gold is wrong.

The 6 structural items are all genuine figure-reading errors: an occluded digit
read as a different digit, the top of a stacked segment reported instead of its
height, an adjacent bar read instead of the asked one, a miscount of printed
values, and two precision misreads of unlabelled bars.

## What it implies, with the caveat that n = 20

1. **Fabrication looks scarce in naturalistic ChartQA errors.** Zero of 20. By
   the rule of three the fabrication rate among these errors is below about 15%
   with 95% confidence, so 317 errors would hold at most a few dozen and
   probably far fewer. Qwen answers with a value or label that is in the figure
   almost every time. This is the risk SPEC Section 9 names ("fabrication
   scarcity in natural data"), and it is what the synthetic arm exists to cover.
   A fabrication probe trained and tested on naturalistic errors alone is
   unlikely to be possible; running more ChartQA splits will not fix this.
2. **About a third of the "errors" may not be errors.** They are scoring
   artifacts or gold mistakes. That is label noise in the binary probe's
   positive class as well as here. If it holds on the full set, the binary probe
   should be re-run with those items removed.
3. **The rubric sends too much to `ambiguous`.** 70% is too high for the labels
   to be useful. Two changes would recover most of it: a separate
   `not_an_error` label (scoring artifact or gold error, to be excluded from
   every probe, including the binary one), and a decision on computation and
   reasoning errors, which are currently outside the taxonomy.

## Decisions needed before labelling continues

- Does the team revise `docs/taxonomy.md` (P2.3 expects at least two rounds)?
- Does the type probe become structural (naturalistic) versus fabrication
  (synthetic), rather than both classes from ChartQA?
- Should the binary probe be re-run once `not_an_error` items are identified?
