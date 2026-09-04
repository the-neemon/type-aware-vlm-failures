"""Activation cache storage budget (open decision 1, TASKS.md Section 7).

Answers one question: for each candidate caching schema, how many bytes does
the activation cache take? The term that dominates is the number of vision
tokens, because it is the only quantity that scales with image resolution, and
it is the difference between a cache that fits on /scratch and one that does
not.

Reads the vendored config snapshots in configs/model_configs/ so the numbers
are reproducible offline. Run:

    python3 src/extract/storage_budget.py
"""

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "model_configs"

BYTES_FP16 = 2
GIB = 1024 ** 3


# --------------------------------------------------------------------------
# Vision token counts
# --------------------------------------------------------------------------

def qwen_smart_resize(height, width, factor, min_pixels, max_pixels):
    """Qwen2.5-VL's dynamic-resolution resize. Mirrors Qwen2VLImageProcessor."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_vision_tokens(height, width, prep, max_pixels=None):
    """Vision tokens Qwen2.5-VL emits for one image, after patch merging."""
    patch = prep["patch_size"]
    merge = prep["merge_size"]
    factor = patch * merge
    h_bar, w_bar = qwen_smart_resize(
        height, width, factor,
        prep["min_pixels"],
        prep["max_pixels"] if max_pixels is None else max_pixels,
    )
    return (h_bar // patch) * (w_bar // patch) // (merge ** 2)


def llava_next_vision_tokens(cfg):
    """LLaVA-NeXT anyres token count at its largest grid pinpoint.

    One base tile plus the anyres grid, each tile contributing
    (image_size / patch_size)^2 tokens, plus one newline token per tile row.
    """
    vc = cfg["vision_config"]
    side = vc["image_size"] // vc["patch_size"]
    per_tile = side ** 2
    best = max(cfg["image_grid_pinpoints"], key=lambda hw: hw[0] * hw[1])
    rows = best[0] // vc["image_size"]
    cols = best[1] // vc["image_size"]
    grid = rows * cols * per_tile
    newlines = rows * side if cfg.get("use_image_newline_parameter") else 0
    return per_tile + grid + newlines


# --------------------------------------------------------------------------
# Storage schemas
# --------------------------------------------------------------------------

# vectors cached per layer, per item
SCHEMAS = {
    "A_pooled": (
        2,
        "mean over vision tokens, last query token",
    ),
    "B_pooled_plus": (
        4,
        "mean + max over vision tokens, last query token, mean over query tokens",
    ),
    "C_unpooled": (
        None,  # resolved per model: every vision token, plus last query token
        "every vision token, plus last query token",
    ),
}


def bytes_per_item(vectors_per_layer, hidden, n_layers):
    return vectors_per_layer * hidden * n_layers * BYTES_FP16


def main():
    qwen = json.loads((CFG / "qwen2_5_vl_7b.config.json").read_text())
    qprep = json.loads((CFG / "qwen2_5_vl_7b.preprocessor.json").read_text())
    llava = json.loads((CFG / "llava_next_mistral_7b.config.json").read_text())

    # LLaVA-NeXT's text_config omits both fields; they fall back to the
    # Mistral-7B-Instruct-v0.2 backbone defaults. Verified against that config.
    llava_hidden, llava_layers = 4096, 32

    models = {
        "Qwen2.5-VL-7B": {
            "hidden": qwen["hidden_size"],
            "layers": qwen["num_hidden_layers"],
            "vis_tokens": None,  # resolution dependent, filled below
        },
        "LLaVA-NeXT-7B": {
            "hidden": llava_hidden,
            "layers": llava_layers,
            "vis_tokens": llava_next_vision_tokens(llava),
        },
    }

    print("## 1. Vision token counts\n")
    print("Qwen2.5-VL uses dynamic resolution, so its token count depends on the")
    print("figure size and on the max_pixels we set in the processor.\n")
    print("| Figure size | max_pixels | Resized | Vision tokens |")
    print("| --- | --- | --- | --- |")
    sweep = [(600, 800), (800, 1000), (1000, 1400), (1080, 1920)]
    caps = [("default (12.8M)", None), ("1.0M", 1_000_000), ("0.5M", 500_000)]
    qwen_tokens_at = {}
    for cap_name, cap in caps:
        for h, w in sweep:
            factor = qprep["patch_size"] * qprep["merge_size"]
            rh, rw = qwen_smart_resize(
                h, w, factor, qprep["min_pixels"],
                qprep["max_pixels"] if cap is None else cap)
            t = qwen_vision_tokens(h, w, qprep, cap)
            print(f"| {h}x{w} | {cap_name} | {rh}x{rw} | {t} |")
            qwen_tokens_at.setdefault(cap_name, []).append(t)
    print()
    print(f"LLaVA-NeXT is fixed-grid anyres: "
          f"{models['LLaVA-NeXT-7B']['vis_tokens']} tokens at its largest "
          f"pinpoint, regardless of figure size.\n")

    # Planning figure: a typical ChartQA chart under a 1.0M pixel cap.
    qwen_typical = qwen_vision_tokens(800, 1000, qprep, 1_000_000)
    models["Qwen2.5-VL-7B"]["vis_tokens"] = qwen_typical
    print(f"Planning number for Qwen: **{qwen_typical} vision tokens** "
          f"(800x1000 figure, max_pixels capped at 1.0M).\n")

    print("## 2. Cache size per schema\n")
    print("fp16 throughout. Layer sets: 8 subsampled layers spanning early,")
    print("middle and late, versus every layer.\n")
    print("| Model | Schema | Layers | Per item | 6k items | 20k items |")
    print("| --- | --- | --- | --- | --- | --- |")

    rows = []
    for mname, m in models.items():
        for sname, (vecs, _desc) in SCHEMAS.items():
            v = vecs if vecs is not None else m["vis_tokens"] + 1
            for lname, nl in (("8", 8), ("all", m["layers"])):
                bpi = bytes_per_item(v, m["hidden"], nl)
                r = (mname, sname, lname, bpi)
                rows.append(r)
                per = (f"{bpi/1024:.0f} KB" if bpi < 1024**2
                       else f"{bpi/1024**2:.1f} MB")
                print(f"| {mname} | {sname} | {lname} | {per} | "
                      f"{bpi*6_000/GIB:.1f} GiB | {bpi*20_000/GIB:.1f} GiB |")

    print("\n## 3. Verdict\n")
    def total(schema, layers):
        return sum(b for (_, s, l, b) in rows if s == schema and l == layers)

    both_bpi = total("B_pooled_plus", "8")
    unpooled_bpi = total("C_unpooled", "8")
    ball_bpi = total("B_pooled_plus", "all")
    print("Planned volume after decisions 2 and 3 is about **6,000 items**:")
    print("roughly 2,800 naturalistic (a 700-error pool plus matched correct")
    print("items, per model) and 3,200 synthetic.\n")
    print(f"- Schema B, ALL layers, both models, 6k items: "
          f"**{ball_bpi*6_000/GIB:.1f} GiB**, "
          f"or {(ball_bpi*6_000 + 1024*1024*6_000)/GIB:.1f} GiB with room to "
          f"re-cache one model.")
    print(f"- Schema B, 8 layers, both models, 6k items: "
          f"**{both_bpi*6_000/GIB:.1f} GiB**.")
    print(f"- Schema C, 8 layers, both models, 6k items: "
          f"**{unpooled_bpi*6_000/GIB/1024:.1f} TiB**, "
          f"a {unpooled_bpi/both_bpi:.0f}x increase. Not viable.")
    print()
    print("Pooling is what makes full-depth caching affordable. Once the vision")
    print("tokens are pooled, going from 8 layers to every layer costs a factor")
    print(f"of {ball_bpi/both_bpi:.1f}x on a base that is already small, so there is no")
    print("reason to subsample layers and inherit the risk of having picked the")
    print("wrong ones.")


if __name__ == "__main__":
    main()
