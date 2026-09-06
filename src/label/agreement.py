"""Inter-annotator agreement and validation against ground truth (TASKS P2.2, P2.4).

Provides:
1. Cohen's kappa calculation between two raters.
2. Ground-truth agreement report for validating against synthetic sets.
3. Disagreement extractor for identifying rubric ambiguities.
"""

from __future__ import annotations

import collections
from typing import Mapping, Sequence

from src.label.annotation import ALLOWED_LABELS


def compute_cohen_kappa(
    rater1_labels: Sequence[str],
    rater2_labels: Sequence[str],
    categories: Sequence[str] = ALLOWED_LABELS,
) -> float:
    """Compute Cohen's kappa coefficient between two sets of annotations.

    Parameters
    ----------
    rater1_labels : Sequence[str]
        Labels from first annotator.
    rater2_labels : Sequence[str]
        Labels from second annotator.
    categories : Sequence[str]
        Possible label categories.

    Returns
    -------
    float
        Cohen's kappa in [-1.0, 1.0]. Returns 1.0 if perfect agreement on single category.
    """
    if len(rater1_labels) != len(rater2_labels):
        raise ValueError(
            f"Label sequences must have identical length, got {len(rater1_labels)} vs {len(rater2_labels)}"
        )
    n = len(rater1_labels)
    if n == 0:
        return 1.0

    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    k = len(categories)
    matrix = [[0] * k for _ in range(k)]

    for l1, l2 in zip(rater1_labels, rater2_labels):
        if l1 not in cat_to_idx or l2 not in cat_to_idx:
            raise ValueError(f"Label not in categories: {l1!r} or {l2!r}")
        matrix[cat_to_idx[l1]][cat_to_idx[l2]] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    r1_margins = [sum(matrix[i][j] for j in range(k)) for i in range(k)]
    r2_margins = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    pe = sum((r1_margins[i] * r2_margins[i]) for i in range(k)) / (n * n)

    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0

    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)


def evaluate_against_ground_truth(
    annotations: Sequence[Mapping[str, str]],
    ground_truth: Mapping[str, str],
) -> dict:
    """Compare annotator records against ground truth failure types.

    Parameters
    ----------
    annotations : Sequence[Mapping[str, str]]
        List of annotation records containing at least 'item_id' and 'label'.
    ground_truth : Mapping[str, str]
        Dictionary mapping item_id -> ground_truth_label.

    Returns
    -------
    dict
        Evaluation summary containing:
        - total_items
        - matched_items
        - agreement_rate
        - cohen_kappa
        - per_class: {label: {precision, recall, count_annotated, count_ground_truth}}
        - disagreements: list of disagreement details
    """
    annotated_ids = [a["item_id"] for a in annotations if a["item_id"] in ground_truth]
    n = len(annotated_ids)
    if n == 0:
        return {
            "total_items": 0,
            "matched_items": 0,
            "agreement_rate": 0.0,
            "cohen_kappa": 0.0,
            "per_class": {},
            "disagreements": [],
        }

    annotator_by_id = {a["item_id"]: a for a in annotations if a["item_id"] in ground_truth}
    ann_labels = [annotator_by_id[item_id]["label"] for item_id in annotated_ids]
    gt_labels = [ground_truth[item_id] for item_id in annotated_ids]

    correct_count = sum(1 for a, g in zip(ann_labels, gt_labels) if a == g)
    agreement_rate = round(correct_count / n, 4)

    # Active categories present in either annotations or ground truth
    all_cats = sorted(list(set(ann_labels) | set(gt_labels)))
    kappa = compute_cohen_kappa(ann_labels, gt_labels, categories=all_cats)

    # Per-class metrics
    per_class = {}
    for cat in all_cats:
        tp = sum(1 for a, g in zip(ann_labels, gt_labels) if a == cat and g == cat)
        pred_pos = sum(1 for a in ann_labels if a == cat)
        actual_pos = sum(1 for g in gt_labels if g == cat)

        prec = round(tp / pred_pos, 4) if pred_pos > 0 else 0.0
        rec = round(tp / actual_pos, 4) if actual_pos > 0 else 0.0
        per_class[cat] = {
            "precision": prec,
            "recall": rec,
            "count_annotated": pred_pos,
            "count_ground_truth": actual_pos,
        }

    # Disagreements list
    disagreements = []
    for item_id in annotated_ids:
        rec = annotator_by_id[item_id]
        ann = rec["label"]
        gt = ground_truth[item_id]
        if ann != gt:
            disagreements.append(
                {
                    "item_id": item_id,
                    "annotator_label": ann,
                    "ground_truth": gt,
                    "question": rec.get("question", ""),
                    "gold_answer": rec.get("gold_answer", ""),
                    "model_answer": rec.get("model_answer", ""),
                    "rationale": rec.get("rationale", ""),
                }
            )

    return {
        "total_items": len(annotations),
        "matched_items": n,
        "agreement_rate": agreement_rate,
        "cohen_kappa": kappa,
        "per_class": per_class,
        "disagreements": disagreements,
    }


def format_evaluation_report(results: dict) -> str:
    """Format evaluation results into a clean markdown table summary."""
    lines = [
        f"**Total evaluated items**: {results['matched_items']}",
        f"**Agreement rate**: {results['agreement_rate'] * 100:.1f}%",
        f"**Cohen's kappa**: {results['cohen_kappa']:.4f}",
        "",
        "### Per-Class Performance",
        "| Class | Ground Truth Count | Annotated Count | Precision | Recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cat, metrics in results["per_class"].items():
        lines.append(
            f"| `{cat}` | {metrics['count_ground_truth']} | {metrics['count_annotated']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} |"
        )

    if results["disagreements"]:
        lines.append("")
        lines.append(f"### Disagreements ({len(results['disagreements'])} items)")
        for d in results["disagreements"]:
            lines.append(
                f"- **{d['item_id']}**: Annotator=`{d['annotator_label']}` vs GT=`{d['ground_truth']}`\n"
                f"  - Question: {d['question']}\n"
                f"  - Gold: {d['gold_answer']} | Model Answer: {d['model_answer']}\n"
                f"  - Rationale: {d['rationale']}"
            )
    else:
        lines.append("")
        lines.append("### Disagreements")
        lines.append("No disagreements found: 100% agreement with ground-truth construction.")

    return "\n".join(lines)
