# Synthetic fabrication pilots: what Qwen does with a bar that is not there

Why these exist: the labelling pilot (`results/labels/pilot/README.md`) found 0
fabrications in 20 naturalistic ChartQA errors, so fabrication examples have to
come from the synthetic arm. A probe trained on "ChartQA structural vs synthetic
fabrication" would learn real-vs-synthetic, and on synthetic data the question
template alone predicts the intended class at AUROC 1.000
(`results/e2-synthetic-is-tautological.md`). So the design is a **within-template
contrast**: the same question type, asked about a missing category, and the
probe separates answers that fabricated from answers that rejected the premise.
Every outcome is scored from the model's actual answer (`src/synth/items.py`).

## Pilot 1 (run 27 Sep, Shrish, gnode090, TAG=synth_pilot)

Generator `src/synth/absent_pairs.py`, config `configs/inference_synth_pilot.yaml`.
300 bar charts, 2 questions each, same words: "What is the value of Mango?"
(a shown bar) and "What is the value of Guava?" (a plausible name from the same
theme, not shown). Values are multiples of 2 or 4 on 0-50/0-100 axes; no zero bars.

**Present bars: 286/300 read correctly (95.3%).** All 14 misses are rounding to a
round number (16 -> 15, 4 -> 5, 64 -> 60), only on 0-100 axes. Too few for a
structural probe.

**Absent names, 300 answers:**

| Answer | Count | Reading |
| --- | --- | --- |
| "0" | 276 | Open question. No bar is zero-height, so nothing was copied; likely "no bar" squeezed into the "single word or phrase" format, but it does assert a value. Not yet decided whether it counts as fabrication. |
| "None" | 9 | Rejection. Only for "What is the value of/for X?"; "How high is the bar for X?" was never rejected (0/88). |
| another number | 15 | Fabrication. 10 exactly copy a similarly named bar (Dubai -> Dublin's 40, Lute -> Flute's 80, Eagle -> Falcon's 80, Camel -> Llama's 18); 3 more are within 1 of one. |

Either way "0" is counted, one side of the fabricated-vs-rejected contrast is too
small (291 vs 9, or 15 vs 285). The summary.json from this run predates the
scorer change and counts "0" as fabricated; re-score its rows with
`src.synth.items.score` to get the current outcomes.

## Pilot 2 (designed 27 Sep, not yet run, TAG=synth_pilot2)

Generator `src/synth/absent_lookalikes.py`, config
`configs/inference_synth_pilot2.yaml`. 300 charts, 6 questions each (1,800
items), three families, each asked once about a shown bar and once about a
missing name in the same words:

| Family | Present | Absent | Absent outcomes |
| --- | --- | --- | --- |
| value | "What is the value of Mango?" | "... of Guava?" | rejected / zero / fabricated / unclear |
| compare | "Which is larger, Mango or Apple?" | "Which is larger, Guava or Apple?" | rejected / fabricated (names Guava) / picked_present (names Apple, arguably the same escape as "0") / unclear |
| neighbor | "Which bar is immediately to the right of Mango?" | "... right of Guava?" | rejected / fabricated (names any bar) / unclear |

Names come in lookalike pairs (Dubai/Dublin, Eagle/Falcon, Lute/Flute); a chart
shows at most one of each pair, and half the absent questions ask for the partner
of a shown bar (`lookalike_of` set), half for an unrelated name
(`lookalike_of: null`). Charts are harder, for more structural misreads: values
are never multiples of 5, axes go to 50, 100 or 200, and half the present compare
questions ask about the two closest bars (`closest_pair`, `gap`).

Expected manifest sha256 on Ada: `41fc27e5...dcf0` (full value in the config).

## Open decisions

1. Does "0" to a value question count as fabrication? Does "picked_present"?
   The neighbor family avoids both escapes, so it is the cleanest test.
2. Scale-up size: the target is about 800 fabrications (TASKS P3.6), so charts
   needed = 800 / (fabrications per chart) from pilot 2's rates.
