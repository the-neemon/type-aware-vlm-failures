# Rubric calibration: 50 Qwen errors re-labelled blind (30 Sep)

Purpose: check that `docs/taxonomy.md` Part A, written on 30 Sep from the 317
Qwen labels of 26 to 28 Sep, reproduces those labels before it is used on
LLaVA-NeXT. If it did not, the LLaVA labels would not be comparable with Qwen's.

Sample: 50 of Qwen's 317 ChartQA test errors, `random.Random(42).sample` over the
annotation pool order. Labelled by two Claude (claude-opus-5-5) subagents, 25
each, from the rubric at commit `400fb12`, seeing only the chart image, question,
gold and model answer, with the original labels withheld.
Labels: `qwen2_5_vl_7b_test.calib50.jsonl`.

## Result

| | |
| --- | --- |
| Label agreement | **47 / 50** |
| Cohen's kappa (5 labels) | **0.917** |
| Label and reason both agree | 45 / 50 |

Sample mix (original labels): not_an_error 16, computation 15, structural 14,
ambiguous 5. No fabrication was drawn (Qwen has one).

## The three disagreements, and what they changed in the rubric

| item | original | new | rule made explicit |
| --- | --- | --- | --- |
| 61138ae9c604 | structural | computation | Trace the model's number: 14.72 is the right average of the wrong series (two blue bars), so structural. |
| 366d51052ec6 | structural | fabrication | A bad estimate of an unlabelled segment (about 76; model 37.5) is structural; fabrication needs the relevant values to be printed. |
| 06789f52ce35 | ambiguous / other | structural | A partly right answer ([2014, 2015] for 2014) is ambiguous / other. |

In each case the original label is the one the rules support, and the rubric now
says so (boundaries table, 30 Sep rows). The same 50 items were not re-labelled
after the change, which would only have tuned the rubric to its own test; the
0.917 stands as the measurement of the rubric as first written.

This is agreement between two applications of one LLM annotator, not the human
agreement check (kappa > 0.6 on 200 items, two people), which is still owed.
