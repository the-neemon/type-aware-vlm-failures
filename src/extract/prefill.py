"""Capture the pre-generation decoder state inside `generate()` (inf.md 5, TASKS P4.1, P4.2).

The hypothesis is about the model's state after it has read the figure and the
question and before it emits an answer token. That is the prefill pass, the
first forward that `model.generate()` runs. Forward hooks on the decoder blocks
keep each block's **first** output per item and ignore the decode steps that
follow, so the activations come from the same call that produced the answer
(TASKS P4.4). There is no second forward pass and no `output_hidden_states`.

Each hooked block's (1, T, hidden) output is pooled on the spot into four
vectors and moved to the CPU as float16, so nothing the size of a sequence
outlives the item:

    vision_mean   mean over the image-token positions
    vision_max    elementwise max over the same positions
    query_last    the final valid prompt position, the pre-generation state
    query_mean    mean over prompt text positions: not image, not special

Layer `i` means the output of decoder block `i`, counting from zero. This is
the residual stream after the block, before the model's final norm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

POSITIONS = ("vision_mean", "vision_max", "query_last", "query_mean")


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PoolMasks:
    """Which prompt positions each pooled vector reads. Built once per item."""
    vision: torch.Tensor      # (T,) bool
    text: torch.Tensor        # (T,) bool
    last: int                 # index of the final valid prompt position

    @property
    def seq_len(self) -> int:
        return len(self.vision)

    @property
    def n_vision(self) -> int:
        return int(self.vision.sum())


def build_pool_masks(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     image_token_id: int, special_ids: Iterable[int]) -> PoolMasks:
    """Masks from the prompt's own token ids, for one unpadded item.

    `image_token_id` and `special_ids` come from the loaded model and tokenizer,
    never from constants. An empty vision mask means the image token id is
    wrong, and every vision vector would then be garbage rather than an error.
    """
    if input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError(f"expected one unpadded item, got input_ids {tuple(input_ids.shape)} "
                         f"and attention_mask {tuple(attention_mask.shape)}")
    ids = input_ids[0].cpu()
    valid = attention_mask[0].cpu().bool()
    special = torch.isin(ids, torch.tensor(sorted(set(special_ids)), dtype=ids.dtype))

    vision = valid & (ids == image_token_id)
    text = valid & ~vision & ~special
    if not vision.any():
        raise ValueError(f"no image tokens (id {image_token_id}) in the prompt")
    if not text.any():
        raise ValueError("no text tokens in the prompt after removing image and special tokens")
    return PoolMasks(vision=vision, text=text, last=int(valid.nonzero()[-1]))


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool(hidden: torch.Tensor, masks: PoolMasks) -> torch.Tensor:
    """(1, T, hidden) block output -> (len(POSITIONS), hidden) float16 on the CPU.

    Pooled in float32 so that a mean over several hundred fp16 positions does
    not round, then stored as float16 like the rest of the cache.
    """
    x = hidden[0].float()
    vision = masks.vision.to(x.device)
    text = masks.text.to(x.device)
    pooled = {
        "vision_mean": x[vision].mean(dim=0),
        "vision_max": x[vision].amax(dim=0),
        "query_last": x[masks.last],
        "query_mean": x[text].mean(dim=0),
    }
    out = torch.stack([pooled[p] for p in POSITIONS]).to(torch.float16).cpu()
    if not torch.isfinite(out).all():
        raise FloatingPointError("non-finite pooled activation; refusing to cache it")
    return out


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def decoder_layers(model, expected: int) -> nn.ModuleList:
    """The language decoder blocks on transformers 4.57.6, verified.

    The same path holds for Qwen2.5-VL (Qwen2_5_VLTextModel) and LLaVA-NeXT
    (LlavaNextModel.language_model, a MistralModel).

    Asserting on the length checks the object that is actually hooked, which is
    what catches a path that moved between library versions.
    """
    layers = model.model.language_model.layers
    if not isinstance(layers, nn.ModuleList) or len(layers) != expected:
        raise RuntimeError(f"expected {expected} decoder layers at "
                           f"model.model.language_model.layers, found {len(layers)}")
    return layers


# What one decoder block returns on transformers 4.57.6, per model family. Named
# per model rather than detected, so an unexpected type raises instead of being
# silently unwrapped.
TUPLE = "tuple"      # Qwen2_5_VLDecoderLayer returns (hidden_states,)
TENSOR = "tensor"    # MistralDecoderLayer (LLaVA-NeXT) returns hidden_states itself
DECODER_OUTPUTS = (TUPLE, TENSOR)


class PrefillCapture:
    """Forward hooks that keep each requested block's prefill output, pooled.

    Per item: `arm(masks)` before `generate()`, `collect()` after it. `collect`
    disarms, so the state of one item cannot leak into the next, and a decoder
    call while disarmed raises rather than being recorded or dropped.

        with PrefillCapture(decoder_layers(model, 28), range(28), 3584, TUPLE) as capture:
            capture.arm(masks)
            model.generate(...)
            acts = capture.collect()      # (28, 4, 3584) float16
    """

    def __init__(self, layers: nn.ModuleList, indices: Sequence[int], hidden_size: int,
                 decoder_output: str = TUPLE):
        if decoder_output not in DECODER_OUTPUTS:
            raise ValueError(f"decoder_output must be one of {DECODER_OUTPUTS}, "
                             f"got {decoder_output!r}")
        self.indices = list(indices)
        self.hidden_size = hidden_size
        self.decoder_output = decoder_output
        self._masks: PoolMasks | None = None
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = [layers[i].register_forward_hook(self._hook(i)) for i in self.indices]

    def __enter__(self) -> PrefillCapture:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def arm(self, masks: PoolMasks) -> None:
        if self._masks is not None:
            raise RuntimeError("capture already armed; the previous item was never collected")
        self._masks = masks
        self._captured = {}

    def collect(self) -> torch.Tensor:
        """(len(indices), len(POSITIONS), hidden) for the item just generated. Disarms."""
        missing = [i for i in self.indices if i not in self._captured]
        captured, self._captured, self._masks = self._captured, {}, None
        if missing:
            raise RuntimeError(f"no prefill captured for decoder layers {missing}")
        return torch.stack([captured[i] for i in self.indices])

    def _hook(self, index: int):
        def hook(module, args, output):
            if self._masks is None:
                raise RuntimeError(f"decoder layer {index} ran while capture was not armed")
            if index in self._captured:
                return                                  # a decode step, not the prefill
            hidden = self._hidden(index, output)
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != self.hidden_size:
                raise ValueError(f"layer {index}: unexpected hidden shape {tuple(hidden.shape)}")
            # The prefill spans the whole prompt. Anything shorter is not the
            # prefill, and anything else would misalign the masks.
            if hidden.shape[1] != self._masks.seq_len:
                raise ValueError(f"layer {index}: first call has sequence length "
                                 f"{hidden.shape[1]}, prompt has {self._masks.seq_len}")
            self._captured[index] = pool(hidden, self._masks)
        return hook

    def _hidden(self, index: int, output) -> torch.Tensor:
        """The block's hidden states, in exactly the form this model family returns."""
        if self.decoder_output == TUPLE:
            if not (isinstance(output, tuple) and len(output) == 1):
                raise TypeError(f"layer {index}: expected a 1-tuple from the decoder "
                                f"block, got {type(output).__name__}")
            return output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"layer {index}: expected a tensor from the decoder "
                            f"block, got {type(output).__name__}")
        return output
