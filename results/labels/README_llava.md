# LLaVA-NeXT ChartQA error labels (30 Sep to 1 Oct)

Labels for all **1,178** errors LLaVA-NeXT (Mistral-7B) made on the ChartQA test
split (2,500 questions, relaxed accuracy 52.9%):
`llava_next_mistral_7b_test.claude.jsonl`. Error statistics:
`llava_next_mistral_7b_test.error_stats.json` (`python -m src.label.error_stats`).

The file is append-only. Where a label was corrected, the original line stays and
a later line with the same `item_id` replaces it, as for every reader in
`src/label/` and `src/probes/`. Each line's `annotator` names who produced it and
the rubric commit it followed.

## How they were made

**Annotator.** Claude (claude-opus-5-5) in Claude Code, the same annotator as the
317 Qwen labels. Each subagent labelled 25 errors in the annotation pool's seed-42
order (`src/label/pool.py`, `src/label/llm_label.py batches`). It saw only the
chart image, the question, the gold answer and the model's answer: no model
identity, no data tables and no other labels. Every batch was checked against the
rubric's labels, reasons and item ids before it was appended
(`llm_label append`).

**Rubric.** `docs/taxonomy.md` Part A. The five-label rubric the Qwen labels
followed had never been committed; it was written on 30 Sep from all 317 Qwen
labels, with worked examples taken from them.

**Calibration first.** Before any LLaVA item, two subagents re-labelled 50 random
Qwen errors blind: 47/50 agreed with the original labels, Cohen's kappa **0.917**
(`calibration/README.md`). The three disagreements each exposed a rule the Qwen
labels applied but the rubric left implicit; those rules were written in.

**Rules made explicit while labelling.** LLaVA answers differently from Qwen: it
garbles printed numbers (486 for 4 863), snaps to gridlines and returns a printed
value where a calculation was asked. Each question this raised was settled from a
Qwen precedent, committed with its date, and the rubric was frozen during every
wave from wave 3 on. The "Final rulings" section (1 Oct) states the six rulings
that apply everywhere.

**Audit.** Batches 1-16 were labelled before several rulings existed, so they
were **re-labelled blind** under the final rubric. On batches 1-8 the two passes
agreed on 180/200 labels (kappa 0.834); the changes were mostly the ones the new
rulings predict (6 fabrication -> structural, 5 structural -> computation). In
batches 17-48, the 75 items a final ruling could change (fabrications, rounding,
colour words, multi-part answers, question_ambiguous, cross-model conflicts) were
reviewed, and all 34 short-answer garble calls were reviewed under ruling 1's
short-answer clause. In all, **55 of 1,178 labels changed** (68 correction lines).

## Result

| Label | LLaVA-NeXT | | Qwen2.5-VL | |
| --- | --- | --- | --- | --- |
| structural | 583 | 49% | 71 | 22% |
| computation | 435 | 37% | 93 | 29% |
| ambiguous | 74 | 6% | 40 | 13% |
| not_an_error | 69 | 6% | 112 | 35% |
| fabrication | 17 | 1.4% | 1 | 0.3% |
| **errors** | **1,178** | | **317** | |

Reasons (LLaVA): not_an_error 39 format_equivalent, 20 valid_reading, 10
gold_error; ambiguous 34 gold_error, 24 question_ambiguous, 15 other, 1
unreadable. 44 gold answers are wrong overall (22 human-written, 22 augmented
questions); 10 answers are the right value on another scale (0.43 for 43).

What differs from Qwen, and why it is plausible rather than a labelling artefact:

1. **LLaVA's errors are real misreads; a third of Qwen's were not errors.** A
   stronger model's "errors" are dominated by scoring artefacts and wrong golds;
   a weaker one's by genuine failures. Only 6% of LLaVA's are not errors.
2. **Structural is LLaVA's largest class** (49% against 22%). The misreads are
   garbled printed values, adjacent bars and years, and gridline snapping (10 000
   for a printed 11 767). The synthetic pilots found the same gridline snapping.
3. **Fabrication stays rare on ChartQA** even for LLaVA: 17 of 1,178. Before the
   audit it was 29; most "made-up" numbers were garbled printed values. ChartQA
   still cannot supply a fabrication class; the synthetic arm has to.

## Consistency checks

On the 272 questions both models got wrong, excluding 32 whose questions
resemble the rubric's worked examples (agreement there is partly built in):

| Check | Result |
| --- | --- |
| Identical answer from both models: same label | **45 / 48** |
| Question whose gold the Qwen labels found wrong: LLaVA label also says gold_error | **30 / 31** |

The remaining disagreements are cases where the **Qwen** label looks
inconsistent with the rubric, not the LLaVA one.

## Nine Qwen labels flagged for the team

`qwen2_5_vl_7b_test.flagged.jsonl`. Found while reconciling the two models; not
changed, because the Qwen labels are the team's record and the paper reports them.
Five would change a label and four only a reason. None touches a structural or
fabrication label, so the Qwen type probes are unaffected; the computation-class
probe would lose one item (the modes question) and not_an_error would gain three.

- one of two modes labelled computation, against the rubric's partly-right rule;
- three questions left open (no age group, candidate or confidence level) where
  Qwen's answer fits one option: `valid_reading` under ruling 6, not
  `question_ambiguous`;
- four gold-status or reason disagreements (gold wrong for one model and valid
  for the other; `unreadable` where the gold is wrong regardless);
- one uncertain `valid_reading` candidate.

## Limits

- **One LLM annotator.** Two applications of it agree well (0.917 on Qwen, 0.834
  across passes), but this is not human agreement. The planned human check (two
  people, 200 items, kappa > 0.6) is still owed and should sample both models.
- Labellers knew nothing of the model, but the rubric's examples are Qwen items,
  so a few questions were recognisable.
- The rubric was refined during labelling. Every change is dated in git, and the
  final audit applied the final version to every batch it could affect.
