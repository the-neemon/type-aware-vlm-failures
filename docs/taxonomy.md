# Synthetic Failure Taxonomy

The observed model answer receives one of three labels. The intended question
type does not determine the label: a model that correctly rejects a missing
category is correct, even when the item targets fabrication.

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
