"""Load LLaVA-NeXT (Mistral-7B) and ask it one question, like src/extract/qwen.py does for Qwen.

The second primary model (SPEC 4.1). Everything that decides what the model is
asked and how the answer is decoded is shared with Qwen rather than copied:
the answer suffix, the seed, greedy decoding with 32 new tokens, the chat
template with the image before the text, and whitespace-only stripping. So the
two models get the same prompt and differ only in the model.

Checked against transformers 4.57.6 and the llava-hf processor files:

- The chat template gives "[INST] <image>\\n{question}{suffix} [/INST]": one
  contiguous block of image tokens starting at position 5, then the question,
  so image-token activations cannot see the question, as for Qwen.
- The processor expands <image> into base + unpadded + image-newline features,
  all as image-token ids, and the model fills exactly those positions, so the
  vision mask covers the newline positions too. 2,242 image tokens for an
  800x557 ChartQA figure (Qwen: 580).
- config.image_token_id is an alias of LLaVA's config.image_token_index.
- MistralDecoderLayer returns a bare tensor, not Qwen's 1-tuple.
- The fast and slow image processors differ (max 0.015-0.030 on normalised
  pixels), as they do for Qwen, so use_fast stays pinned to True. The slow
  path would also need sentencepiece for the tokenizer, which is not pinned.

Imports torch lazily, so the constants are usable on a machine with no GPU.
"""

from __future__ import annotations

# Shared with Qwen on purpose: the same prompt, seed and decoding for both models.
from src.extract.qwen import (  # noqa: F401  (re-exported)
    ANSWER_SUFFIX, MAX_NEW_TOKENS, SEED, build_inputs, generate, generate_from_inputs,
)

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
MODEL_KEY = "llava_next_mistral_7b"   # hashed into every item_id (inf.md 3.1)

DECODER_OUTPUT = "tensor"             # MistralDecoderLayer returns hidden_states itself
# The pipeline floor, set the way Qwen's is: roughly 0.7 of the published score
# (Qwen: 0.60 against ~87%). LMMs-Eval (Zhang et al. 2024, arXiv:2407.12772,
# Table 1) reports LLaVA-NeXT-Mistral-7B at 38.8 on ChartQA; the often-quoted
# ~55 is LLaVA-NeXT-Vicuna-7B (54.8 there). Catches a broken prompt or pipeline,
# never a reason to tune the prompt.
MIN_ACCURACY = 0.27


def config_checks(entry: dict) -> list[tuple[str, object, object]]:
    """(name, config value, code value) for this model's configs/activations.yaml entry.

    Nothing beyond the checks every model gets: LLaVA-NeXT has no pixel budget
    to set. Its preprocessing is recorded by `processor_settings` instead.
    """
    return []


def processor_settings(processor) -> dict:
    """Image-preprocessing settings for the run record; they change activations."""
    ip = processor.image_processor
    return {"image_grid_pinpoints": ip.image_grid_pinpoints,
            "image_size": ip.size,
            "image_crop_size": ip.crop_size,
            "image_do_pad": getattr(ip, "do_pad", None),
            "patch_size": processor.patch_size,
            "vision_feature_select_strategy": processor.vision_feature_select_strategy}


def load(model_id: str = MODEL_ID):
    """Returns (model, processor) on the frozen config: fp16, SDPA, fast processor.

    Same reasoning as qwen.load: the card is the RTX 2080 Ti, where bf16 is
    emulated and FlashAttention-2 does not run.
    """
    import torch
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; refusing to fall back to CPU")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    # Without these the processor emits a single <image> token and the model
    # fails deep inside generate() with a feature-count mismatch.
    if processor.patch_size is None or processor.vision_feature_select_strategy is None:
        raise SystemExit("LLaVA-NeXT processor lacks patch_size or "
                         "vision_feature_select_strategy; the processor files are incomplete")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, attn_implementation="sdpa", device_map="auto")
    got = next(model.parameters()).dtype
    if got != torch.float16:
        raise SystemExit(f"asked for float16, model loaded as {got}")
    return model, processor
