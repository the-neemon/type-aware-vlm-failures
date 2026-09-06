# P2.2 Annotation Tool Validation Report

**Date**: 2026-09-02 (Validated 2026-09-06)  
**Task**: P2.2 Labelling Tool Build & Validation  
**Annotator**: Shrish Kadam  
**Dataset**: 20 synthetic bar-chart failure items generated from `data/synthetic_bar_charts` (Seed 42)  
**Rubric Reference**: `docs/taxonomy.md`  

---

## 1. Tool Implementation Summary

The annotation tool was implemented in `src/label/` in accordance with the project constraints:
- **No Web App**: Built as a streamlined CLI (`src/label/annotate.py`) with single-keypress hotkeys (`s`, `f`, `a`, `q`) via `msvcrt` on Windows (with `stdin` fallback for automation/scripted pipelines) and a compact desktop GUI (`--gui` via Python standard `tkinter`).
- **Complete Blinding**: The annotator view (`AnnotationItem`) strictly excludes model identity (`model_name`, `model_id`) and any existing/ground-truth labels to preserve the integrity of inter-annotator agreement (Cohen's kappa).
- **Per-Item Disk Persistence**: Every annotation record is appended to disk and immediately flushed with `os.fsync` after each item, guaranteeing that a mid-session crash costs at most one item.
- **Resumability**: Automatically identifies already-annotated `item_id`s in the target output file and resumes from the first unannotated item.

---

## 2. Validation Against Synthetic Ground Truth

Twenty synthetic failure instances were selected and annotated blind to test the interface and validate the definitions in `docs/taxonomy.md`:
- **10 Structural failure items**:
  - Direct bar value read errors (misreading height or confusing with an adjacent bar)
  - Reversed pairwise bar comparisons (`A` vs `B`)
  - Miscalculated bar height differences
- **10 Fabrication failure items**:
  - Asserting specific numerical values for absent categories (e.g., category `F`)
  - Asserting unsupported values far outside the visible chart axis (e.g., claiming `145` on a `0-100` axis)

### Results

```
**Total evaluated items**: 20
**Agreement rate**: 100.0%
**Cohen's kappa**: 1.0000

### Per-Class Performance
| Class | Ground Truth Count | Annotated Count | Precision | Recall |
| --- | --- | --- | --- | --- |
| `fabrication` | 10 | 10 | 1.000 | 1.000 |
| `structural` | 10 | 10 | 1.000 | 1.000 |

### Disagreements
No disagreements found: 100% agreement with ground-truth construction.
```

---

## 3. Rubric Assessment (`docs/taxonomy.md`)

- **Structural class**: The definition in `docs/taxonomy.md` ("answer is supported by the figure but the model reads the wrong value, label, comparison, path, or relationship") cleanly covered all height misreads and reversed visual relationships.
- **Fabrication class**: The definition ("model asserts a value, category, node, edge, or label that the figure does not support") unequivocally distinguished queries on absent categories and out-of-range values.
- **Ambiguous class**: The escape hatch exists and is accessible via the `a` key; neither item required dropping or coercing during this synthetic validation.
- **Conclusion**: The rubric in `docs/taxonomy.md` is robust and ready for the upcoming pilot annotation on naturalistic items.
