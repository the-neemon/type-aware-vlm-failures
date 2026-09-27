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

## Pilot 2 (run 27 Sep, Shrish, TAG=synth_pilot2)

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
Shrish's copy matched it; the run passed validation.

### What Qwen answered

Rejections: **6 of 900** absent questions, all "None" to "What is the value of/for
X?". Qwen answers nearly every question about a missing bar.

| Absent template | Outcomes (rescored with the current scorer) |
| --- | --- |
| value | "0" 218, other number 76, "None" 6 |
| compare | names the shown bar 263, names the absent bar or its lookalike 37 |
| neighbor | names a bar of the chart 300 |

Lookalikes drive the made-up answers:

- value: a number other than 0 for 66/143 lookalike names vs 10/157 unrelated
  ones, and 57 of the 76 numbers are the lookalike bar's value (within 5%).
- compare: 24 of the 30 "names the absent bar" answers come when its lookalike is
  the larger bar; the 7 answers naming a third bar are all the lookalike ("Osaka"
  for "Oslo or Seoul?"). The scorer now counts those as fabricated.
- neighbor: for lookalike names, the answer is the bar right of the lookalike 47
  times, or the lookalike itself 40 times, of 140.

So when Qwen fabricates content here, it mostly binds the missing name to the
nearest-looking bar that is there.

Present questions: 95.8% right. compare 300/300, neighbor 294/300, value 268/300.
The 32 misreads concentrate on large, sparse axes: 0-200 with ticks every 40 gives
21/52 (40%); 0-100 gives 6/101; 0-50 gives 0/101.

### Probes (`src/probes/run_synth.py`; report `results/probes/synth_pilot2_qwen2_5_vl_7b.json`)

Figure-level split (180/60/60 charts), (position, L2, layer) chosen on validation,
test scored once, 95% CI by figure bootstrap. Text baseline = logistic on
question-text features (asked name, other name, phrasing).

| Contrast | Test pos/neg | Probe test AUROC [CI] | Where | Text | Vision only |
| --- | --- | --- | --- | --- | --- |
| value: absent vs shown name | 62/62 | 0.999 [0.998, 1.0] | query_last L4 | 0.506 | 0.500 |
| compare: absent vs shown | 62/62 | 0.991 [0.976, 1.0] | query_last L4 | 0.515 | 0.500 |
| neighbor: absent vs shown | 62/62 | 1.000 [1.0, 1.0] | query_last L5 | 0.487 | 0.500 |
| value, absent: other number vs "0" | 14/46 | 0.983 [0.949, 1.0] | query_last L27 | 0.704 | 0.727 |
| same, lookalike names only | 11/18 | 1.000 [1.0, 1.0] | query_last L22 | 0.795 | 0.737 |
| compare, absent: names absent/lookalike vs shown | 6/56 | 0.994 [0.973, 1.0] | query_last L19 | 0.655 | 0.360 |
| value, shown: misread vs right | 9/53 | 0.933 [0.849, 0.992] | query_last L19 | 0.607 | 0.803 |

How to read it:

1. **Qwen represents that the asked-about name is missing, from layer 2 on, and
   answers anyway.** Validation AUROC is 0.51 to 0.57 at layers 0-1 and 0.99 to
   1.00 from layer 2 in all three families, and it holds for lookalike names
   (test 0.999, 0.980, 1.000 on lookalike-only absent items). The controls behave:
   the text baseline is at chance (every name is a bar on some charts and missing
   on others), and the vision positions score exactly 0.500, because image tokens
   precede the question and cannot see it. Caveat: "is this word printed on the
   chart" is a property of the input, and on these items nearly every absent
   question is answered wrongly, so this probe detects the condition that causes
   fabrication; it is the signal a controller would act on.
2. **"Other number vs 0" and "absent vs shown bar named" peak late** (layers 19
   to 27; early layers sit near the text and lookalike-flag baselines). These
   largely read out the answer about to be produced, not an early warning.
3. **Misread prediction is real but small-n**: 0.933 with 9 test positives. Vision
   alone gets 0.803 because hard charts (0-200, sparse) are visible in the image.
4. **Cross-type check (E3 in miniature):** the absence probe's score does not
   separate misreads from right answers among shown-bar value questions (test
   AUROC 0.46, 9 misread vs 53 right). Consistent with fabrication risk and
   misread risk being different directions, but far too few misreads to claim it.

Limits: one pilot, one split, 60 test charts; several contrasts have under 15
test positives; synthetic charts only.

## Open decisions

1. Does "0" to a value question count as fabrication? Does "picked_present"?
   The neighbor family avoids both escapes, so it is the cleanest test.
2. The planned "fabricated vs rejected" probe is not possible under the standard
   prompt: Qwen rejects 6 of 900. The contrast that works is absent vs shown name
   within one template, above.
3. Scale-up: fabrications are no longer scarce (every absent neighbor question
   yields one). Misreads are: at pilot 2's mix, 800 would need ~7,500 value
   questions; on 0-200 sparse axes alone (40%) about 2,000.
