"""Load Qwen2.5-VL-7B and ask it one question, exactly as inf.md Section 4 specifies.

The one place these settings live. Every job that queries the model (the
inference run, E4's repairs) goes through here, because cross.md 3.3 is right
that two copies of them can drift: a repair re-queried with a different dtype,
attention implementation or processor is measuring the configuration change,
not the repair. Same hazard as two definitions of `item_id`.

Imports torch lazily, so the constants are usable on a machine with no GPU.

Activation caching (inf.md 5) registers its forward hooks on the returned model
before calling `generate()`; the hooks then fire inside the same forward pass
that produces the answer, which is what TASKS P4.4 requires.
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_KEY = "qwen2_5_vl_7b" # hashed into every item_id (inf.md 3.1)
MAX_PIXELS = 1_000_000      # configs/activations.yaml
SEED = 42
MAX_NEW_TOKENS = 32         # configs/inference.yaml
REASK_TEMPERATURE = 0.7     # I_reask only; everything else is greedy
ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."   # inf.md 4.1

# What src/extract/cache_chartqa.py needs to know about this model family.
DECODER_OUTPUT = "tuple"    # Qwen2_5_VLDecoderLayer returns (hidden_states,) on 4.57.6
# Below this relaxed accuracy the prompt or the pipeline is broken (inf.md 8.3):
# the published ChartQA number for Qwen2.5-VL-7B is about 87%.
MIN_ACCURACY = 0.60


def config_checks(entry: dict) -> list[tuple[str, object, object]]:
    """(name, config value, code value) for this model's configs/activations.yaml entry."""
    return [("activations max_pixels", entry["max_pixels"], MAX_PIXELS)]


def processor_settings(processor) -> dict:
    """Image-preprocessing settings for the run record; they change activations."""
    import importlib.metadata

    return {"qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
            "max_pixels": MAX_PIXELS,
            "image_processor_max_pixels": getattr(processor.image_processor, "max_pixels", None)}


def load(model_id: str = MODEL_ID):
    """Returns (model, processor) on the frozen config: fp16, SDPA, fast processor.

    fp16 and SDPA are fixed rather than chosen by hardware. The card is frozen
    as the RTX 2080 Ti (Turing), where bf16 is emulated rather than refused and
    FlashAttention-2 does not run, so picking by capability would only ever
    matter on a card this project must not use.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; refusing to fall back to CPU")
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=MAX_PIXELS, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, attn_implementation="sdpa", device_map="auto")
    # a misspelled dtype argument silently loads fp32 and dies as an OOM that
    # looks like a hardware problem (inf.md 4.3)
    got = next(model.parameters()).dtype
    if got != torch.float16:
        raise SystemExit(f"asked for float16, model loaded as {got}")
    return model, processor


def build_inputs(model, processor, image, prompt: str):
    """Chat-template one image plus one prompt, on the device holding the embeddings."""
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    # with device_map="auto" the model spans both cards; accelerate's hooks move
    # activations across the shard boundary from the embedding device
    return inputs.to(model.get_input_embeddings().weight.device)


def generate(model, processor, image, prompt: str, *, sample: bool = False,
             max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """One answer. Greedy unless `sample`; decoded answer is whitespace-stripped only."""
    inputs = build_inputs(model, processor, image, prompt)
    return generate_from_inputs(model, processor, inputs, sample=sample,
                                max_new_tokens=max_new_tokens)


def generate_from_inputs(model, processor, inputs, *, sample: bool = False,
                         max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """`generate` on inputs already built, for callers that need them too.

    Activation caching builds its pooling masks from `inputs` before the call,
    and its hooks capture the prefill inside this same `model.generate`.
    """
    import torch

    torch.manual_seed(SEED)
    kw = dict(do_sample=True, temperature=REASK_TEMPERATURE) if sample else dict(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **kw)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
