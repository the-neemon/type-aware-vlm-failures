"""Prepare 20 synthetic validation items with ground-truth failure types.

Selects 20 concrete items from the synthetic bar charts:
- 10 Structural errors (misread bar heights, reversed comparisons, misread differences)
- 10 Fabrication errors (asserting values for absent categories, asserting out-of-range values)

Outputs:
- data/synthetic_validation_items.jsonl (blind items for the annotation tool)
- data/synthetic_validation_ground_truth.json (ground truth failure types by construction)
"""

import json
import pathlib
import random


def build_validation_items():
    synth_dir = pathlib.Path("data/synthetic_bar_charts")
    manifest_file = synth_dir / "manifest.jsonl"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Synthetic manifest not found at {manifest_file}")

    figs = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                figs.append(json.loads(line))

    items = []
    ground_truth = {}
    rng = random.Random(42)

    # 1. 10 Structural errors from first 10 charts
    for i in range(10):
        fig = figs[i]
        fig_id = fig["figure_id"]
        img_path = str(pathlib.Path("data/synthetic_bar_charts") / fig["image_path"])
        cats = fig["categories"]
        vals = fig["values"]
        val_map = dict(zip(cats, vals))

        if i < 4:
            # read_value structural error: reads another category's value
            cat = cats[0]
            gold = val_map[cat]
            other_val = val_map[cats[1]]
            item_id = f"synth_{fig_id}_structural_read"
            items.append({
                "item_id": item_id,
                "figure_id": fig_id,
                "image_path": img_path,
                "question": f"What is the value of category {cat}?",
                "gold_answer": str(gold),
                "model_answer": str(other_val),
            })
            ground_truth[item_id] = "structural"

        elif i < 7:
            # compare_bars structural error: reverses comparison
            c1, c2 = cats[0], cats[1]
            larger = c1 if val_map[c1] > val_map[c2] else c2
            smaller = c2 if larger == c1 else c1
            item_id = f"synth_{fig_id}_structural_compare"
            items.append({
                "item_id": item_id,
                "figure_id": fig_id,
                "image_path": img_path,
                "question": f"Which category is larger, {c1} or {c2}?",
                "gold_answer": larger,
                "model_answer": smaller,
            })
            ground_truth[item_id] = "structural"

        else:
            # bar_difference structural error: misreads one bar
            c1, c2 = cats[0], cats[1]
            diff = abs(val_map[c1] - val_map[c2])
            # model uses c3 instead of c2
            wrong_diff = abs(val_map[c1] - val_map[cats[2]])
            if wrong_diff == diff:
                wrong_diff = diff + 10
            item_id = f"synth_{fig_id}_structural_diff"
            items.append({
                "item_id": item_id,
                "figure_id": fig_id,
                "image_path": img_path,
                "question": f"What is the absolute difference between {c1} and {c2}?",
                "gold_answer": str(diff),
                "model_answer": str(wrong_diff),
            })
            ground_truth[item_id] = "structural"

    # 2. 10 Fabrication errors from charts 10 to 19
    for i in range(10, 20):
        fig = figs[i]
        fig_id = fig["figure_id"]
        img_path = str(pathlib.Path("data/synthetic_bar_charts") / fig["image_path"])
        cats = fig["categories"]
        vals = fig["values"]

        if i < 17:
            # absent_category fabrication: claims category F has value 55
            missing_cat = "F"
            item_id = f"synth_{fig_id}_fabrication_absent"
            items.append({
                "item_id": item_id,
                "figure_id": fig_id,
                "image_path": img_path,
                "question": f"What is the value of category {missing_cat}?",
                "gold_answer": "not present",
                "model_answer": str(vals[0]),  # Fabricates that absent cat has value
            })
            ground_truth[item_id] = "fabrication"

        else:
            # out-of-range fabrication: claims a value unsupported by axis (145 on 0-100 axis)
            cat = cats[0]
            item_id = f"synth_{fig_id}_fabrication_outofrange"
            items.append({
                "item_id": item_id,
                "figure_id": fig_id,
                "image_path": img_path,
                "question": f"What is the value of category {cat}?",
                "gold_answer": str(vals[0]),
                "model_answer": "145",  # Fabricated unsupported number well above axis_max=100
            })
            ground_truth[item_id] = "fabrication"

    # Shuffle deterministically to prevent pattern-guessing during annotation
    paired = list(zip(items, [ground_truth[it["item_id"]] for it in items]))
    rng.shuffle(paired)

    shuffled_items = [p[0] for p in paired]
    shuffled_gt = {p[0]["item_id"]: p[1] for p in paired}

    # Write blind items (strictly no ground_truth or target_failure_type fields)
    out_items_path = pathlib.Path("data/synthetic_validation_items.jsonl")
    out_items_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_items_path, "w", encoding="utf-8") as f:
        for item in shuffled_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write ground truth reference
    out_gt_path = pathlib.Path("data/synthetic_validation_ground_truth.json")
    with open(out_gt_path, "w", encoding="utf-8") as f:
        json.dump(shuffled_gt, f, indent=2)

    print(f"Generated {len(shuffled_items)} synthetic validation items at {out_items_path}")
    print(f"Ground truth stored at {out_gt_path}")


if __name__ == "__main__":
    build_validation_items()
