# Failure Taxonomy

Two rubrics live here. **Part A** labels ChartQA errors by hand (or by an LLM
annotator), five labels. **Part B** is the synthetic arm, where outcomes are
scored automatically from the model's answer (`src/synth/items.py`).

In both, the label describes the observed answer, not the intended question type.

---

# Part A: ChartQA errors (five labels)

Written 30 Sep 2026 from the 317 Qwen2.5-VL labels in
`results/labels/qwen2_5_vl_7b_test.claude.jsonl` (labelled 26 to 28 Sep), which
followed these rules but whose rubric was never committed. Every rule and example
below is taken from those labels. Use it unchanged for every model, so labels are
comparable across models.

## What the annotator sees

The chart image, the question, the gold answer and the model's answer. Nothing
else: not the model's identity, not the ChartQA data tables, not other labels.
An item is only in the pool because relaxed accuracy (5% numeric tolerance, exact
string match otherwise) scored it wrong.

## Record format

One JSON line per item, appended, later lines win:

```json
{"item_id": "...", "label": "structural", "reason": null,
 "rationale": "one or two sentences with the numbers read off the chart",
 "annotator": "...", "timestamp": "ISO 8601"}
```

`reason` is required for `not_an_error` and `ambiguous`, and `null` for the
other three. The rationale states what the chart shows, what the correct answer
is and where the model's answer comes from, with the actual values.

## Deciding, in order

1. **Is the model's answer actually right?** If so: `not_an_error`, whatever
   relaxed accuracy said.
2. **Can the correct answer be determined from the chart?** If not, or if the
   gold is wrong and so is the model: `ambiguous`.
3. **Where does the model's answer come from?**
   - from reading the figure wrongly: `structural`;
   - from correctly read values, handled wrongly: `computation`;
   - from nowhere in the figure: `fabrication`;
   - cannot be told apart: `ambiguous` (`other`).

## `not_an_error`: the answer is right

| reason | when | examples |
| --- | --- | --- |
| `format_equivalent` | Same answer, different form. | 0.95 for 95 (or the reverse, fraction vs percent); "1:2" for 0.5; "2:1" for 2; "213k" for 213; 151000 for 151 (thousands); "increase" for "increasing"; "Facebook Messenger" for "Facebook Messenger*" (footnote marker); "Australia, Italy" for "[Australia, Italy]"; "Fox 4" for "Fox" (value appended); "Gregs" for "Greggs" (typo); "Grade 10" for "Girls grade 10" when the question already restricts to girls; "2004 to 2005" for "2005" when both name the same step. |
| `format_equivalent` (colours) | A colour name for the same mark: "black" for a near-black "Dark blue" bar, "Blue" for "light blue", "Dark blue" for "Navy blue", "Red" for a rust "orange" line, "Green" for "Teal Blue". Only when it identifies the same mark. |
| `valid_reading` | The question admits the model's reading and the model answered it correctly. | A 30% cell is row Compatibility x column Very concerned, model named the row and gold the column; no year given, model used the latest year; "how many more times" read as a ratio (model) vs a difference (gold); the inverse ratio when the question does not fix the order; the total market (model) vs the domestic market (gold) for "companies in the market"; "who received the highest percentage" answered with the person, gold gives the value. |
| `gold_error` | The chart supports the model and contradicts the gold. | The 2021 bar is printed 5 014, the model's answer; gold 5 857 is the 2019 bar. "Second largest": model names the second bar, gold the first. |

## `ambiguous`: no single answer can be checked

| reason | when | examples |
| --- | --- | --- |
| `question_ambiguous` | The question has no single correct answer from this chart (names no group or year where the chart splits them, garbled, offers only wrong options), and the model's answer does not match a valid reading either. | "Male smokers in England in 2019" with only age groups shown; "does the line increase or decrease" for a flat line; a three-way ratio asked for as one number. |
| `gold_error` | The gold is wrong **and** the model is wrong too. | Gold ratio 1.058 matches no pair of bars (50/48 = 1.042); model 0.25 matches none either. |
| `unreadable` | The needed value cannot be read at the chart's resolution. | An unlabelled point on a flat line on a -100% to 700% axis; two near-identical teal shades when counting colours. |
| `other` | Anything else outside the taxonomy: misread vs arithmetic slip cannot be told apart; the answer needs outside knowledge or data the chart lacks (relative incidence from absolute counts); the answer is incomplete, e.g. one of two correct years. | Model 0.01 for gold 0 traces to no printed value; "which country was a party to the Treaty of Versailles". |

If the model is right under one reading and gold under another, that is
`not_an_error` / `valid_reading`, not `ambiguous`.

## `structural`: the figure was read wrongly

A more accurate look at the same figure would recover the answer.

- **Wrong mark**: the adjacent bar or year (the 2017 bar for 2018); the wrong
  segment of a stacked bar; the whole bar instead of the asked series (Iran has
  the longest total bar, Saudi Arabia the largest oil segment); the top of a
  stacked segment instead of its height; a value matched in the wrong year.
- **Wrong series by colour or legend**: the grey series for blue, men's values
  for women's, the navy line above the blue one, the wrong slice by colour.
- **Imprecise read** of an unlabelled bar, segment or point (9.3 read as 10.3
  just below the 10% gridline), or a misread digit (45 707 read as 45 307).
- **Miscount of visible items**: bars, sectors, years, points, including counts
  against a threshold whose values are printed ("how many bars above 30%": the
  five printed values 34, 43, 40, 43, 36, model says 4).
- **Misjudged shape or position**: which line is higher, where a line peaks, the
  steepest slope, a reversed comparison of two bars of similar length.
- **Axes confused**: the y-axis title given for "what does the x-axis represent".

## `computation`: correctly read values, handled wrongly

Every input is printed or clearly readable, and the operation is wrong.

- **Arithmetic** on printed values: sums, differences, means, medians, products,
  a shifted decimal (0.2139 for 2.14).
- **Wrong operation**: a ratio of levels where the question asks for a ratio of
  changes; a sum where a difference is asked; the maximum where the median is.
- **Wrong selection by a stated criterion**: "largest light-blue value" answered
  with the leftmost; "last four countries" shifted by one row; the wrong pair of
  printed values.
- **Incomplete aggregation**: "80 and above" answered with the 80-89 slice only;
  a difference left unsubtracted.
- **Wrong comparison or ranking of printed values**: which printed value is
  largest or second largest; the larger of two printed changes; the wrong
  direction ("least peaceful" on an axis where higher is less peaceful).
- **Counts or criteria needing a derived quantity**: "how many bars exceed twice
  the smallest"; "how many categories represent at least comfortable".

A computed answer that matches no combination of printed values is still
`computation`, not `fabrication`: the model attempted the operation.

**Trace the model's number before choosing between `structural` and
`computation`.** Try the asked operation on the neighbouring marks, the other
series and the adjacent years. If the answer is the right operation on the wrong
marks, it is `structural`. Example: the shortest grey (16.92) and light-blue
(14.58) bars average 15.75; the model's 14.72 is (14.67 + 14.77) / 2, two blue
bars, so the grey series was misidentified.

## Boundaries

| Case | Label |
| --- | --- |
| Count of visible items or of printed values past a stated threshold, wrong | `structural` |
| Count that needs a derived quantity first (twice the smallest, at least X) | `computation` |
| Wrong series picked by colour or legend | `structural` |
| Wrong item picked by a stated criterion (leftmost instead of largest) | `computation` |
| "Which is largest" among **unlabelled** bars, wrong | `structural` |
| "Which is largest" among **printed** values, wrong | `computation` |
| Misread input then used in arithmetic (the misread explains the answer) | `structural` |
| Right inputs, wrong arithmetic | `computation` |
| Lookup answer that matches nothing, where the needed values are **printed** | `fabrication` |
| Wrong estimate of an **unlabelled** mark, however far off | `structural` |
| Computed answer that matches no combination | `computation` |
| Right operation applied to the wrong marks or series | `structural` |
| Partly right: one of two correct items, or the correct item plus a wrong extra | `ambiguous` / `other` |
| Cannot tell a misread from an arithmetic slip | `ambiguous` / `other` |
| All inputs clearly printed and the arithmetic result is off | `computation` |

The rows on printed vs unlabelled marks, wrong marks, partly right answers and
clearly printed arithmetic were made explicit on 30 Sep, after a 50-item blind
calibration against the Qwen labels (47/50 agreement, kappa 0.917). Each states
a rule those labels already applied; the three disagreements are in
`results/labels/calibration/README.md`.

## `fabrication`: content the figure does not support

The model states a value, label or category for a lookup question that appears
nowhere in the figure and follows from no reading of it. Only when the values
the question needs are printed: a bad estimate of an unlabelled bar or segment
is `structural` even when far off (an unlabelled segment of about 76, the model
says 37.5). Example: slices are 35,
40, 13 and 11, "important" is 35 + 40 = 75, and the model says 55, which is no
slice and no sum of slices. Rare on ChartQA (1 of Qwen's 317), because ChartQA
questions are about what the chart shows.

---

# Part B: synthetic charts (scored automatically)

## Structural

Use `structural` when the answer is supported by the figure but the model reads
the wrong value, label, comparison, path, or relationship. A more accurate read
of the existing figure could recover the answer.

1. The chart shows B at 50. Asked for B, the model answers 45. This is an axis or
   height misread; 45 is not treated as fabricated merely because it is wrong.
2. A is 48 and B is 50. Asked which is larger, the model says A. Both entities
   exist, but their visual relationship was reversed.
3. The graph contains A-C-D-B. Asked for the shortest path from A to B, the
   model answers A-E-B. The task is structural if E exists elsewhere and the
   model traced the wrong edges; it is fabrication if E does not exist.

## Fabrication

Use `fabrication` when the model asserts a value, category, node, edge, or label
that the figure does not support. Looking more precisely at the same content
cannot recover the claimed information.

1. The chart contains A, B, and C. Asked for D, the model states that D is 40.
   D is absent, so its claimed value is fabricated.
2. The displayed values range from 10 to 80. The model states that a bar is 140
   without a valid computation producing 140. The value is unsupported.
3. The graph has no node Z. The model claims that Z connects directly to A. Both
   the node and relationship are fabricated.

**"0" for a missing category is not fabrication** (decided 28 Sep). Asked for D
when the chart has no D, a bare "0" is a refusal expressed as a number: no bar
is zero-height on our synthetic charts, so nothing was read or copied, and Qwen
says "0" mostly when no bar resembles D (141 of 157 unrelated names in synthetic
pilot 2). It is still a wrong answer, since a reader would take D to exist with
value 0, so it is counted as an error of its own kind and kept out of the
fabrication class. A non-zero number for D is fabrication, as in example 1.

## Neither or ambiguous

Use `ambiguous` when the gold answer is questionable, the figure is unreadable,
the question has multiple valid interpretations, the response does not reveal
which failure occurred, or the mistake falls outside this taxonomy. Do not force
these items into either failure class; exclude them from type-probe training and
retain them for an audit count.
