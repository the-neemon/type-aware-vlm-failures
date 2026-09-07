"""Check the analytic vision-token counts against the real Qwen processor.

The storage budget in results/storage_budget.md is built on token counts that
src/extract/storage_budget.py derives analytically from the config, by
reimplementing smart_resize. Nothing has confirmed that reimplementation
matches what Qwen2VLImageProcessor actually does, and an off-by-one in the
merge or a different rounding rule would shift the whole budget table.

This script closes that gap. Run it once the environment exists and the Qwen
checkpoint is downloaded (TASKS.md P0.2, P0.3):

    python3 scripts/verify_vision_tokens.py path/to/chartqa/figures/*.png

Exits non-zero on any mismatch. A mismatch means storage_budget.py is wrong
and configs/activations.yaml needs recomputing before caching starts.
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract.storage_budget import qwen_vision_tokens  # noqa: E402

HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CFG = ROOT / "configs" / "model_configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=pathlib.Path)
    ap.add_argument("--max-pixels", type=int, default=1_000_000,
                    help="must match models.qwen2_5_vl_7b.max_pixels in "
                         "configs/activations.yaml")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    prep = json.loads((CFG / "qwen2_5_vl_7b.preprocessor.json").read_text())
    image_token_id = json.loads(
        (CFG / "qwen2_5_vl_7b.config.json").read_text())["image_token_id"]

    # use_fast pinned to match configs/activations.yaml and the inference
    # path; fast and slow differ in pixels, though not in token counts.
    processor = AutoProcessor.from_pretrained(
        HF_ID, max_pixels=args.max_pixels, use_fast=True)

    print(f"{'figure':<28} {'size':>12} {'analytic':>9} {'actual':>7}  ok")
    print("-" * 68)

    failures = []
    for path in args.images:
        img = Image.open(path).convert("RGB")
        w, h = img.size

        predicted = qwen_vision_tokens(h, w, prep, args.max_pixels)

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What is the value of the tallest bar?"},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        actual = int((inputs["input_ids"] == image_token_id).sum())

        ok = predicted == actual
        if not ok:
            failures.append((path.name, predicted, actual))
        print(f"{path.name[:28]:<28} {f'{w}x{h}':>12} {predicted:>9} "
              f"{actual:>7}  {'yes' if ok else 'NO'}")

    print()
    if failures:
        print(f"MISMATCH on {len(failures)} of {len(args.images)} figures.")
        print("src/extract/storage_budget.py does not model the processor "
              "correctly. Fix it, regenerate results/storage_budget.md, and "
              "recheck configs/activations.yaml before caching anything.")
        return 1

    print(f"All {len(args.images)} figures match. The storage budget stands.")
    print("Report the observed token range so the planning number in "
          "configs/activations.yaml can be replaced with a measured one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
