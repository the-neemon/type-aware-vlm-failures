"""The E4 interventions: what gets re-asked, with which image and prompt (P6.1 to P6.4).

Pure functions on images and strings, so everything here is testable without a
GPU. `run_matrix.py` feeds the results to the model. Spec: cross.md.

| Name         | Image                          | Prompt              | Decoding       |
| ------------ | ------------------------------ | ------------------- | -------------- |
| `I_0`        | not re-asked; original answer  |                     |                |
| `I_reask`    | original                       | original            | sampled, T=0.7 |
| `I_upsample` | whole figure, upscaled         | original            | greedy         |
| `I_crop`     | target bars + y-axis, upscaled | original            | greedy         |
| `I_verify`   | original                       | name evidence first | greedy, longer |
| `I_abstain`  | not re-asked; scored in E5     |                     |                |

Decisions taken 25 September (TASKS decisions 5 and 6):

- **`I_0` stays the pre-registered baseline**: the original wrong answer, so its
  recovery is zero by definition. That alone would let any repair look good for
  free, so **`I_reask`** is added: the same question again, sampled rather than
  greedy (greedy would just repeat the answer). It separates "this repair works"
  from "asking twice works", and is reported alongside the pre-registered test,
  not instead of it.
- **`I_crop` is question-conditioned, never attention-conditioned** (SPEC 4.3).
  It keeps the y-axis strip, because a crop of a bar without its scale cannot
  be read. It needs bar geometry, which only synthetic charts record, so on
  ChartQA it is not applicable until a text-matched crop exists (that needs
  OCR). **`I_upsample`** is the floor and runs everywhere: without it, a gain
  from cropping cannot be told apart from simply having more pixels.
- **The manipulated variable is where the pixels go, not how many there are.**
  `max_pixels` stays at the pinned 1,000,000 for every repair, and both
  `I_upsample` and `I_crop` enlarge their image to fill that same budget. So the
  two see the same number of vision tokens, and `I_crop` minus `I_upsample`
  isolates localisation from resolution. (cross.md suggests raising
  `max_pixels` for the crop instead; that would give the crop more tokens than
  its own floor and fold resolution back into the comparison.)
- When the question names a category that is not on the chart, there is nothing
  to crop to and `I_crop` shows the whole figure. That is the fabrication case,
  and the hypothesis predicts cropping should not help it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from PIL import Image

# The prompt and pixel budget come from the one module that loads the model, so
# a repair asks the question exactly as the original run did (cross.md 3.3).
from src.extract.qwen import ANSWER_SUFFIX, MAX_PIXELS

VERIFY_SUFFIX = (
    "\nFirst name the part of the chart that supports your answer, such as a bar, "
    "a label or an axis value. Then give the final answer on its own line, "
    "starting with 'Answer:', using a single word or phrase."
)

MAX_SCALE = 4.0            # beyond this, upscaling only blurs
CROP_MARGIN = 12           # pixels either side of the target bars

BASELINE, REASK, UPSAMPLE, CROP, VERIFY, ABSTAIN = (
    "I_0", "I_reask", "I_upsample", "I_crop", "I_verify", "I_abstain")
REQUERIED = (REASK, UPSAMPLE, CROP, VERIFY)


@dataclass(frozen=True)
class Query:
    image: Image.Image
    prompt: str
    sample: bool = False
    max_new_tokens: int = 32
    note: str = ""


@dataclass(frozen=True)
class Figure:
    """What a repair may know about the figure. Geometry is optional."""
    image: Image.Image
    categories: tuple[str, ...] = ()
    bar_boxes: tuple[tuple[int, int, int, int], ...] = ()
    plot_box: tuple[int, int, int, int] | None = None
    extra: dict = field(default_factory=dict)

    @property
    def has_geometry(self) -> bool:
        return bool(self.bar_boxes) and self.plot_box is not None


def upscale_to_budget(image: Image.Image, max_pixels: int = MAX_PIXELS,
                      max_scale: float = MAX_SCALE) -> Image.Image:
    """Enlarge as far as the pixel budget allows, never shrinking."""
    w, h = image.size
    scale = min(math.sqrt(max_pixels / (w * h)), max_scale)
    if scale <= 1.0:
        return image
    new = (max(1, math.floor(w * scale)), max(1, math.floor(h * scale)))
    return image.resize(new, Image.Resampling.LANCZOS)


def question_targets(question: str, categories: tuple[str, ...]) -> list[int]:
    """Indices of the chart's categories that the question names.

    Case-sensitive whole-word match, which is enough for the synthetic templates
    ("category A", "A or B"). Returns [] when the named category is absent.
    """
    return [i for i, c in enumerate(categories)
            if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", question)]


def question_crop(fig: Figure, targets: list[int],
                  margin: int = CROP_MARGIN) -> Image.Image:
    """The y-axis strip beside the target bars, full height, as one image.

    Full height keeps the category labels under the bars and the top of the
    scale. The y-axis strip is everything left of the plot area, which holds the
    tick labels; without it a bar's height cannot be read.
    """
    img = fig.image
    w, h = img.size
    axis_right = fig.plot_box[0] + 2                     # include the spine
    left = max(axis_right, min(fig.bar_boxes[i][0] for i in targets) - margin)
    right = min(w, max(fig.bar_boxes[i][2] for i in targets) + margin)

    strip = img.crop((0, 0, axis_right, h))
    slab = img.crop((left, 0, right, h))
    out = Image.new(img.mode, (strip.width + slab.width, h), "white")
    out.paste(strip, (0, 0))
    out.paste(slab, (strip.width, 0))
    return out


def parse_verified(text: str) -> tuple[str, bool]:
    """Pull the final answer out of an I_verify response. Returns (answer, parsed).

    An unparseable response counts as not recovered and is reported as such,
    rather than guessing which line was meant to be the answer.
    """
    for line in reversed(text.strip().splitlines()):
        m = re.match(r"\s*\**\s*answer\s*\**\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("*").strip(), True
    return text.strip(), False


def build_query(intervention: str, question: str, fig: Figure) -> Query | None:
    """The re-query for one repair, or None when it does not apply to this figure."""
    image = fig.image.convert("RGB")
    if intervention == REASK:
        return Query(image, question + ANSWER_SUFFIX, sample=True)
    if intervention == UPSAMPLE:
        return Query(upscale_to_budget(image), question + ANSWER_SUFFIX)
    if intervention == CROP:
        if not fig.has_geometry:
            return None
        targets = question_targets(question, fig.categories)
        if not targets:
            return Query(upscale_to_budget(image), question + ANSWER_SUFFIX,
                         note="named category absent; whole figure shown")
        crop = question_crop(Figure(image, fig.categories, fig.bar_boxes, fig.plot_box),
                             targets)
        return Query(upscale_to_budget(crop), question + ANSWER_SUFFIX)
    if intervention == VERIFY:
        return Query(image, question + VERIFY_SUFFIX, max_new_tokens=96)
    raise ValueError(f"{intervention!r} is not re-queried")
