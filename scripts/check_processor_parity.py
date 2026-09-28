"""Measure whether the fast and slow image processors agree, pixel for pixel.

Run this whenever `transformers` is upgraded, before re-caching anything.

Background. transformers 4.57 loads `Qwen2VLImageProcessorFast` by default and
warns that it "may produce slightly different outputs". P0.10 (7 Sep) measured
it: vision-token counts are identical, but the preprocessed pixels are not, with
a worst absolute difference of 0.030 on normalised values across five ChartQA
figures. That is far too small to change an answer and entirely large enough to
change a cached activation, which is the quantity this project reports on. So
`processor_use_fast` is frozen in configs/activations.yaml rather than left to
whatever the installed library happens to default to.

This script is the check behind that decision. It does not need a GPU and it
does not need the model weights, only the processor files. It compares the two
IMAGE processors directly, since the pixels are the question; for LLaVA-NeXT
that also avoids the slow tokenizer, which needs the unpinned sentencepiece.

LLaVA-NeXT, measured offline 28 Sep: the fast and slow image processors differ
too (max 0.015 to 0.030), so its use_fast is pinned for the same reason.

    python scripts/check_processor_parity.py <figures...>
    python scripts/check_processor_parity.py --model llava <figures...>
"""

import argparse
import pathlib
import sys

HF_IDS = {"qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
          "llava": "llava-hf/llava-v1.6-mistral-7b-hf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=pathlib.Path)
    ap.add_argument("--model", choices=sorted(HF_IDS), default="qwen")
    ap.add_argument("--max-pixels", type=int, default=1_000_000,
                    help="must match models.qwen2_5_vl_7b.max_pixels in "
                         "configs/activations.yaml (Qwen only)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    # Qwen's pixel budget; LLaVA-NeXT has none, its anyres grid decides instead
    kw = {"max_pixels": args.max_pixels} if args.model == "qwen" else {}
    fast = AutoImageProcessor.from_pretrained(HF_IDS[args.model], use_fast=True, **kw)
    slow = AutoImageProcessor.from_pretrained(HF_IDS[args.model], use_fast=False, **kw)
    print(f"{args.model}: {type(fast).__name__} vs {type(slow).__name__}")

    print(f"{'figure':<28} {'patches':>8} {'max|diff|':>11} {'mean|diff|':>11}  identical")
    print("-" * 74)

    worst = 0.0
    mismatched = 0
    for path in args.images:
        image = Image.open(path).convert("RGB")
        out = {}
        for name, proc in (("fast", fast), ("slow", slow)):
            out[name] = proc(images=[image], return_tensors="pt")

        a = out["fast"]["pixel_values"].float()
        b = out["slow"]["pixel_values"].float()
        if a.shape != b.shape:
            print(f"{path.name[:28]:<28} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
            mismatched += 1
            continue

        diff = (a - b).abs()
        mx, mn = diff.max().item(), diff.mean().item()
        identical = bool(torch.equal(a, b))
        worst = max(worst, mx)
        if not identical:
            mismatched += 1
        print(f"{path.name[:28]:<28} {a.shape[0]:>8} {mx:>11.6f} {mn:>11.6f}  "
              f"{'yes' if identical else 'no'}")

    print()
    print(f"worst absolute pixel difference: {worst:.6f}")
    if mismatched:
        print(f"{mismatched} of {len(args.images)} figures differ between the two "
              "processors.")
        print("This is expected on transformers 4.57 and is why "
              "`processor_use_fast` is pinned in configs/activations.yaml.")
        print("It is NOT a failure. It becomes one only if the pinned value is "
              "ignored somewhere, so grep for from_pretrained calls that omit "
              "use_fast.")
    else:
        print("The two processors agree exactly on these figures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
