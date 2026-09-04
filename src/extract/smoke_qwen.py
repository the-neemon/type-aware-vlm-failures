"""P0.5 smoke test: one ChartQA figure, one question, one answer string.

The point is not the answer. The point is that the whole path works on Ada:
venv, pinned library versions, weights on node-local /scratch, two 2080 Tis
holding one 7B model between them, the Qwen processor, and greedy decoding.

Three Ada-specific choices are load-bearing and are asserted, not assumed:

  fp16, not bf16.  The 2080 Ti is Turing (sm_75) and has no bf16 tensor cores.
  Qwen2.5-VL's reference code runs bf16 on Ampere and later. Running bf16 here
  is emulated and slow where it works at all, so the project is fp16 and that
  is recorded next to the library versions.

  sdpa, not flash_attention_2.  FlashAttention-2 requires sm_80 or later.

  device_map="auto" over 2 GPUs.  fp16 weights are ~16.6 GB against 11 GB of
  VRAM per card, so the model is sharded. Quantising to fit one card is not an
  option: this project probes internal activations, and quantisation perturbs
  exactly the quantity being measured.

    python src/extract/smoke_qwen.py --sample-dir <dir> --out <result.json>
"""

import argparse
import json
import pathlib
import platform
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=pathlib.Path, required=True,
                    help="output of scripts/fetch_chartqa_sample.py")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-pixels", type=int, default=1_000_000,
                    help="matches models.qwen2_5_vl_7b.max_pixels in "
                         "configs/activations.yaml")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # ---- hardware sanity ---------------------------------------------------
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; check --gres in the sbatch")
    n_gpu = torch.cuda.device_count()
    caps = [torch.cuda.get_device_capability(i) for i in range(n_gpu)]
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory
                     for i in range(n_gpu)) / 1024**3
    print(f"gpus: {n_gpu} x {names[0]}  cap={caps[0]}  total VRAM={total_vram:.1f} GiB")

    if len(set(caps)) != 1:
        raise SystemExit(f"mixed GPU capabilities on one node: {caps}")

    # Turing (7.5) has no bf16 tensor cores. Choose dtype from the hardware
    # rather than copying the reference snippet, and record what was chosen.
    dtype = torch.float16 if caps[0][0] < 8 else torch.bfloat16
    attn = "sdpa" if caps[0][0] < 8 else "flash_attention_2"
    print(f"dtype={dtype}  attn_implementation={attn}")

    if total_vram < 17:
        raise SystemExit(
            f"only {total_vram:.1f} GiB VRAM visible; fp16 weights need ~16.6 GiB. "
            "Request --gres=gpu:2.")

    # ---- one item ----------------------------------------------------------
    manifest = json.loads((args.sample_dir / "manifest.json").read_text())
    item = manifest[0]
    img_path = args.sample_dir / item["figure_id"]
    image = Image.open(img_path).convert("RGB")
    question = item["question"]
    print(f"\nfigure  : {item['figure_id']} ({image.width}x{image.height})")
    print(f"question: {question}")
    print(f"gold    : {item['gold']}")

    # ---- model -------------------------------------------------------------
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, attn_implementation=attn, device_map="auto")
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s; device_map spans "
          f"{len(set(str(p.device) for p in model.parameters()))} devices")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = inputs.to(model.device)

    # Free diagnostic for Naman's P0.4 budget: the real vision-token count.
    image_token_id = model.config.image_token_id
    n_vision = int((inputs["input_ids"] == image_token_id).sum())
    print(f"vision tokens: {n_vision}   total input tokens: {inputs['input_ids'].shape[1]}")

    # ---- greedy decode (P1.2: reproducible, no sampling) -------------------
    torch.manual_seed(0)
    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                   do_sample=False)
    gen_only = generated[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
    dt = time.time() - t0

    print("\n" + "=" * 60)
    print(f"ANSWER: {answer!r}")
    print("=" * 60)
    print(f"(gold was {item['gold']!r}; not scored here, that is P1.1)")

    record = {
        "figure_id": item["figure_id"],
        "figure_size": [image.width, image.height],
        "question": question,
        "gold": item["gold"],
        "answer": answer,
        "vision_tokens": n_vision,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "generate_seconds": round(dt, 2),
        "model": args.model,
        "dtype": str(dtype),
        "attn_implementation": attn,
        "max_pixels": args.max_pixels,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                     "seed": 0},
        "env": {
            "node": platform.node(),
            "gpus": names,
            "compute_capability": [list(c) for c in caps],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
