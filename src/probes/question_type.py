"""Tag a ChartQA question as arithmetic or retrieval, by a fixed keyword rule.

Computation errors can only happen on questions that ask for an operation over
read values (a difference, a sum, an average, a ratio), and the wording that
asks for it is in the activations from the first layer. A computation probe
trained against all correct answers can therefore score well by detecting the
question type alone (results/chartqa_type_probes.md: layer 0 already gives 0.865
for LLaVA). Restricting both classes to arithmetic questions removes that route;
restricting misreads and correct answers to retrieval questions does the same for
the structural probe.

The rule was fixed on 1 Oct 2026 before any restricted probe was run, from the
wording of the question only (never the label or the answer). It is
deliberately simple and errs towards "retrieval": ranking questions ("the third
largest", "which year was highest") stay retrieval, because they need a reading
of the chart rather than an operation on read values.
"""

import re

ARITHMETIC = re.compile(
    r"\b("
    r"differen\w*|sum|summ\w*|add(ed|ing)?|addition|"
    r"average|avg|mean|median|mode|"
    r"ratio|times|product|multipl\w*|divid\w*|division|subtract\w*|minus|plus|"
    r"gap|"
    r"(increase|decrease|grow|growth|change|rise|drop|fall)\w* by|"
    r"by how (much|many)|how (much|many) (more|less|fewer|higher|lower|greater|smaller)"
    r")\b",
    re.IGNORECASE)


def question_type(question: str) -> str:
    """'arithmetic' when the question asks for an operation on values, else 'retrieval'."""
    return "arithmetic" if ARITHMETIC.search(question) else "retrieval"
