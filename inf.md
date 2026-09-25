# inf.md: Qwen2.5-VL-7B inference and activation caching on Ada

**Audience:** whoever implements the ChartQA inference run (TASKS P1.2, P1.3,
P1.6, P1.7 and P4.1 to P4.4). You do not need the project's history to follow
this, but you do need to follow the output contract in Section 3 exactly,
because three other pieces of work are already written against it.

**Owner of this spec:** Yash (workstream A). Ask before deviating from
Sections 3, 4 or 5; everything else is implementation freedom.

---

## 1. What you are building, and why the shape matters

One job that runs Qwen2.5-VL-7B over the ChartQA **test** split and emits two
coupled artefacts: a row per question recording what the model answered and
whether it was right, and a cache of the model's internal activations for that
same question.

The project's claim is that a model's internal state before it generates
predicts *what kind* of mistake it is about to make. So the activations and the
answer must come from the same forward pass. If they are produced by two
separate runs they can silently diverge, and the entire result becomes
uninterpretable without anything failing loudly. TASKS P4.4 is the rule; Section
5 below is how to satisfy it.

**Scope for this run, decided 25 September:**

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` only. Not LLaVA-NeXT yet. |
| Data | ChartQA **test** split only: `test_human` + `test_augmented`, 2,500 questions over **1,509 distinct figures** |
| Output | predictions for all 2,500; activations for all 2,500 |
| Expected runtime | 6 to 8 hours at batch size 1 on two 2080 Tis |

Cache activations for **every** item, not only the errors. The binary probe
(correct vs incorrect) needs both classes, and the whole cache is only about
1.9 GiB (Section 6), so filtering buys nothing and costs a re-run.

---

## 2. Prerequisites, already done for you

**Environment.** Built and pinned. Do not create a new one and do not upgrade
anything:

```bash
source $HOME/envs/vlmfail/bin/activate
```

If it is missing on your account, rebuild it from the committed pins, **from
inside a job and not on the head node** (the head node runs an older glibc than
the compute nodes, so a venv built there will not import):

```bash
srun -p u22 --constraint=2080ti --exclude=gnode077 --gres=gpu:1 \
     -c 8 --mem=32G -t 01:00:00 bash scripts/setup_env.sh
```

Pinned versions are exact, not floors: `torch==2.8.0+cu126`,
`transformers==4.57.6`, `qwen-vl-utils==0.0.14`, `accelerate==1.14.0`,
Python 3.10.12. Activations shift between library versions, so a different
`transformers` build is a correctness bug, not a convenience difference.

**Data.** Run once on a compute node. It is idempotent:

```bash
bash scripts/download_chartqa.sh
```

Lands at `$HOME/data/ChartQA`. Home directories on Ada are `drwx------`, so you
cannot read anyone else's copy and need your own. Load it with the existing
loader rather than parsing JSON yourself:

```python
from src.eval.chartqa import load_chartqa
data = load_chartqa(os.path.expanduser("~/data/ChartQA"))
items = data["test_human"] + data["test_augmented"]
# each item: {figure_id, question, gold, split, source}
```

**One thing to be aware of when you concatenate them.** `test_human` covers 625
figures and `test_augmented` covers 987, but **103 figures appear in both**, so
the union is 1,509 distinct figures and not 1,612. Measured on the downloaded
copy, so it is a property of ChartQA and not of our ingest.

It does not affect this job, which processes every question regardless. It
matters downstream: the probe's train/val/test split must be **figure-level**,
and those 103 figures have to land wholly in one split or the figure leaks
across the boundary and inflates every probe number. Carrying `figure_id` in
both output files (Section 3) is what lets the split code enforce that. Just do
not assume "human" and "augmented" are already figure-disjoint groups.

**Scoring.** Use the existing implementation. ChartQA is scored with *relaxed
accuracy*, which allows 5 percent tolerance on numeric answers and exact match
on strings. Scoring with exact match marks correct answers as errors, and those
mislabelled items flow straight into the "structural" class and poison every
probe downstream:

```python
from src.eval.relaxed_accuracy import is_correct
correct = is_correct(gold, prediction)     # tolerance defaults to 0.05
```

---

## 3. The output contract. This part is not negotiable.

Three artefacts must share one join key. Downstream, the labelling step reads
the predictions, the probe reads the activations, and they have to line up row
for row after arbitrary filtering and re-ordering. **Position in a file is not a
key**: it breaks the moment anything is filtered, sorted or resumed.

### 3.1 The join key

**Do not write your own.** One already exists and is committed, and the
labelling pipeline is built on it:

```python
from src.label.pool import item_id

iid = item_id(model, figure_id, question, source, occurrence)
```

For reference, that function is:

```python
def item_id(model: str, figure_id: str, question: str, source: str, occurrence: int) -> str:
    raw = f"{model}\x1f{source}\x1f{figure_id}\x1f{question}\x1f{occurrence}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]
```

Import it rather than copying it. An earlier draft of this spec defined a
near-identical function with `|` as the separator and 16 hex characters instead
of 12. Same inputs, same algorithm, **different output string**, so the two sets
of ids would not have joined and nothing would have raised an error: the pool
would simply have come out empty, or worse, half-populated. `\x1f` is the ASCII
unit separator and cannot occur inside a question or a filename, which is why it
is the better choice and why its version wins.

If a shared location for this helper is preferred later, move it once and have
both sides import from the new home. Two definitions is the failure mode; which
module holds the one definition does not matter.

**Three details that feed the hash, so they must match exactly:**

- `model` must be the literal string **`"qwen2_5_vl_7b"`**, the key already used
  in `configs/activations.yaml`. Not the HuggingFace id, not a display name, and
  not the shorter `qwen`. Note that `src/label/pool.py`'s own usage example
  shows `--source qwen=...`; when it is run against this output it must be
  invoked as `--source qwen2_5_vl_7b=results/predictions/qwen2_5_vl_7b_test.jsonl`,
  or every id changes.
- `source` is the literal ChartQA source, `"human"` or `"augmented"`. It is
  required because some source files repeat the same figure/question pair.
- `occurrence` is the zero-based occurrence of that exact source/figure/question
  tuple in dataset order.
- `figure_id` is the image filename with no directory, exactly as
  `ChartQAItem.figure_id` gives it.
- `question` is the raw question string, **unmodified**. Do not strip, lowercase
  or normalise it. The labelling step recomputes this hash from the manifest, so
  any cleaning silently breaks the join.

This five-field tuple is also the resumability key in Section 7, so compute it once per
item and reuse it.

**Already verified compatible:** `src/label/pool.py` builds its annotation pool
by calling `read_manifest()` on this file, and `read_manifest` reads its five
named fields and ignores the rest. So the extra `item_id`, `model`, `split` and
`source` fields specified below are safe to add and do not need a separate file.

### 3.2 Predictions: `results/predictions/qwen2_5_vl_7b_test.jsonl`

JSON Lines, one object per question, all 2,500 of them including the correct
ones. This is the single source of truth that labelling and caching both key off
(TASKS P1.6).

```json
{
  "item_id": "a3f9c2b1d8e04f67",
  "model": "qwen2_5_vl_7b",
  "figure_id": "two_col_40391.png",
  "question": "How many food items are shown in the bar graph?",
  "gold": "14",
  "prediction": "14",
  "correct": true,
  "split": "test",
  "source": "human"
}
```

`source` is `"human"` or `"augmented"` and is carried through because the two
sub-splits have very different difficulty and the accuracy sanity check in
Section 8 is reported per sub-split.

The existing `src/eval/manifest.py` defines the core five fields. Extend it
rather than replacing it, keeping `figure_id`, `question`, `gold`, `prediction`,
`correct` with those names and meanings.

### 3.3 Activations: `$HOME/activations/qwen2_5_vl_7b/test.npz`

One compressed `.npz` holding, for each cached layer, one array per pooling
position, plus the row order.

```python
np.savez_compressed(
    path,
    item_ids=np.array(item_ids, dtype="<U16"),   # row order, length N
    figure_ids=np.array(figure_ids, dtype=object),  # parallel, for the bootstrap
    **{f"L{layer}_{pos}": arr for ...}           # each arr: (N, 3584) float16
)
```

Requirements:

- `item_ids[i]` names the item in row `i` of **every** array in the file. The
  probe reads a layer array and a label vector and assumes they are aligned; the
  `item_ids` array is what makes that checkable rather than assumed.
- `figure_ids` is stored alongside because `evaluate_at()` in
  `src/probes/sweep.py` clusters its bootstrap by figure, not by item, and it
  should not have to re-join to the predictions file to get it.
- Arrays are `float16`. Never `float32`: it doubles the cache for no probe
  benefit.
- Key naming is `L{layer}_{position}`, for example `L14_vision_mean`.

### 3.4 Why this shape

`src/probes/sweep.py` takes `Activations = dict[layer_index, np.ndarray]` of
shape `(n_items, hidden)` plus a flat binary `y` array and a parallel
`figure_ids` sequence. The `.npz` above loads into exactly that with a dict
comprehension and no reshaping. Do not invent a per-item file layout: 2,500
small files is slow on Ada's NFS and makes the alignment check impossible.

---

## 4. Inference configuration

### 4.1 The prompt, which is load-bearing

ChartQA gold answers are short, usually one number or one phrase, and relaxed
accuracy compares against them literally. Left to itself, Qwen2.5-VL answers
chart questions in prose. In our smoke test it replied

> "The bar graph shows 15 different food items. These are: 1. Lamb 2. Corn ..."

against a gold answer of `14`. That scores as wrong even when the model knows
the answer, and every such item becomes a fake "error" that a human or a judge
then has to label. So the prompt must constrain the format:

```python
SUFFIX = "\nAnswer the question using a single word or phrase."

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": question + SUFFIX},
]}]
text = processor.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
```

This is the standard ChartQA evaluation prompt for this model family, not a
tuned one. Record the exact string in `configs/inference.yaml` next to the
decoding settings. Strip whitespace from the decoded answer and nothing else:
no lowercasing, no punctuation stripping, no unit removal. `is_correct` already
handles numeric parsing, and silently "cleaning" predictions changes the error
set.

### 4.2 Decoding (TASKS P1.2)

Every downstream label refers to one specific answer string, so the answer has
to be reproducible.

```python
torch.manual_seed(42)          # project-wide seed, TASKS 4.5
model.generate(**inputs,
               do_sample=False,       # greedy, no temperature, no top_p
               max_new_tokens=32)
```

Write these into `configs/inference.yaml` alongside the prompt, the model id and
the resolved library versions.

### 4.3 Hardware, and three things that fail quietly

The card is frozen for the project as the **RTX 2080 Ti**: Turing, compute
capability 7.5, 11 GiB of VRAM. Three consequences, none of which raise an
error:

1. **A 7B model does not fit on one card.** fp16 weights are about 16.6 GiB
   against 11.26 GiB of VRAM. Request `--gres=gpu:2` and load with
   `device_map="auto"`. Do not quantise to fit one card: quantisation perturbs
   the activations this project measures.
2. **Turing has no bf16 tensor cores.** The reference code for Qwen2.5-VL uses
   bf16. Here bf16 returns finite numbers and is merely emulated, so nothing
   crashes and everything is subtly different. Use **`float16`**.
3. **FlashAttention-2 needs sm_80.** Use `attn_implementation="sdpa"`.

Also pin the image processor. `transformers` 4.57 loads the *fast* processor by
default and warns that it "may produce slightly different outputs". It does:
measured over five ChartQA figures, no figure came out bit-identical and the
worst absolute pixel difference was 0.030 on normalised values. Token counts are
unaffected. That is far too small to change an answer and easily large enough to
change a cached activation, so it is frozen:

```python
processor = AutoProcessor.from_pretrained(
    MODEL, max_pixels=1_000_000, use_fast=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.float16, attn_implementation="sdpa", device_map="auto")
```

`max_pixels=1_000_000` is pinned in `configs/activations.yaml`. Under it, real
ChartQA figures cost 180 to 630 vision tokens, with the common 800x557 size at
580 (measured, `results/p0_10_vision_tokens.md`).

Note that `transformers` renamed `torch_dtype` to `dtype` partway through the
4.x line. On 4.57.6 `dtype` is correct, but assert the loaded dtype afterwards,
because passing the wrong name silently loads fp32 and dies as an OOM that looks
like a hardware problem:

```python
assert next(model.parameters()).dtype == torch.float16
```

Working reference for all of the above: `src/extract/smoke_qwen.py`, which runs
one item end to end and already handles dtype selection, the device map and the
VRAM check.

---

## 5. Activation caching

### 5.1 What to capture, and when

The hypothesis is about the **pre-generation** state: the model's internal
representation after it has read the figure and the question, before it emits
its first answer token. That is the prefill forward pass, the one `generate()`
does first.

Capture it with forward hooks on the decoder layers, registered before
`generate()` and recording **only their first invocation** per item. That gives
the prefill activations from the same pass that produced the answer, which is
what TASKS P4.4 requires.

Do **not** pass `output_hidden_states=True` to `generate()`. It retains hidden
states for every generated step, which is 32 times more memory than needed and
will OOM on an 11 GiB card. Do **not** run a separate forward pass to collect
states either: it is a second pass and re-introduces exactly the coupling risk
P4.4 exists to prevent.

### 5.2 Verify the hooks fire where you think (TASKS P4.2)

A silently misindexed hook produces a clean null result that looks like a
scientific finding. The module path to the decoder layers is a `transformers`
internal and has changed across versions, so **discover it and assert, rather
than hardcoding a path from a blog post**:

```python
# find the decoder layer list, then prove it is the right one
layers = model.model.language_model.layers   # verify on 4.57.6; adjust if needed
assert len(layers) == 28, f"expected 28 decoder layers, got {len(layers)}"
```

Assert on `len(layers)` as the primary check, because it tests the object you
are actually hooking. If you also want to cross-check the config, note that
`num_hidden_layers` (28) and `hidden_size` (3584) are **top level** in this
model's `config.json`, not nested under `text_config`, and where the loaded
config object exposes them has moved between `transformers` versions. So
discover it rather than hardcoding a path:

```python
cfg = model.config
n = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
assert n == 28
```

The vendored copies in `configs/model_configs/` are the ground truth for these
numbers and need no network access to read.

In every hook, assert the tensor shape before pooling:

```python
def hook(module, args, output):
    h = output[0] if isinstance(output, tuple) else output
    assert h.ndim == 3 and h.shape[-1] == 3584, f"unexpected shape {h.shape}"
```

Layer indices in the output file must mean "the output of decoder layer *i*",
counting from 0. State which convention you used in a comment; an off-by-one
here is invisible later.

### 5.3 The four pooled positions

Pooling is what makes this affordable. Storing raw vision tokens instead would
need terabytes. For each cached layer, store four vectors of length 3584:

| Key | Definition |
| --- | --- |
| `vision_mean` | mean over positions where `input_ids == model.config.image_token_id` |
| `vision_max`  | elementwise max over those same positions |
| `query_last`  | the hidden state at the **final prompt position**, the pre-generation state |
| `query_mean`  | mean over the prompt's non-image, non-special text positions |

Build the masks from `input_ids` and assert that the vision mask is non-empty
and that its size falls in the measured 180 to 630 range. An empty vision mask
means the image token id is wrong and every `vision_*` vector would be garbage
rather than an error.

Run at **batch size 1**. Batching needs left padding, and padded positions
silently contaminate a mean pool unless every mask is handled exactly right. At
roughly 10 seconds per item the whole split is about 7 hours, which fits a
single 12 hour walltime. Correctness is worth more than the speedup here.

### 5.4 Which layers

`configs/activations.yaml` currently says `layers: all` (all 28), with four
positions, `float16`.

> **Open, and Yash will pin it before the run.** Do not hardcode a layer list.
> Read it from `configs/activations.yaml`, accept either `all` or an explicit
> list, and record the resolved list in the run config written next to the
> output. If the answer has not arrived by the time you are ready to launch,
> cache all 28: the full cache is only about 1.9 GiB and dropping layers later
> is free, whereas adding them back costs the whole 7 hour run.

---

## 6. Storage, and why it goes where it goes

Per item: 28 layers x 4 positions x 3584 dims x 2 bytes = **784 KiB**.
For 2,500 items: **about 1.9 GiB**.

Ada has two kinds of storage and they behave very differently. The full map is
in `results/ada_filesystem.md`; the two rules that matter here:

- **`/scratch` is node-local and it is purged.** It is bind-mounted to the same
  directory as `/tmp`, and `tmpreaper` deletes anything untouched for 7 days. A
  job on another node sees none of it. Use it as working space during the job.
- **`$HOME` is the only shared, writable, durable path**, at 30 GiB per user.

So write activations to `/scratch` while the job runs, and **stage them back to
`$HOME` before the job exits** (TASKS P0.9). If you skip this the cache
fragments across whichever nodes the scheduler picked and no probe can be
trained on it. `scripts/ada_env.sh` already provides `stage_out`, which fails
the job loudly and names the stranding node if the copy does not happen.

Check your quota before launching. 1.9 GiB of activations plus 1.1 GiB of
ChartQA against a 30 GiB quota is comfortable, but not if the account is already
near the limit:

```bash
quota -s | tail -3
```

---

## 7. Resumability (TASKS P1.7)

A 7 hour run will not always survive the queue. Jobs must write results
incrementally and skip work already done on restart, keyed by `item_id`.

- Append each prediction row to the `.jsonl` as it is produced, then `flush()`
  and `os.fsync()`. A crash costs one item, not the run.
- On start, read the existing predictions file and build the set of completed
  `item_id`s, then process only the remainder.
- Activations cannot be appended to an `.npz`. Write per-shard `.npy` files into
  `/scratch` as you go (for example one shard per 250 items), then assemble the
  single `.npz` in a finalisation step once every item is present. The shard
  files carry their own `item_ids` so assembly is order-independent.
- The finalisation step **must assert** that the assembled `item_ids` exactly
  match the set in the predictions file, with no duplicates and none missing.
  Fail loudly if not.

---

## 8. Acceptance criteria

The run is done when all of these hold. Check them; do not assume them.

1. **Counts.** `results/predictions/qwen2_5_vl_7b_test.jsonl` has exactly 2,500
   rows, and `item_id` is unique across them.
2. **Alignment.** The `item_ids` in the `.npz` are the same set, in a known
   order, with no duplicates:
   ```python
   assert set(npz["item_ids"]) == {r["item_id"] for r in rows}
   assert len(npz["item_ids"]) == len(set(npz["item_ids"]))
   for k in npz.files:
       if k.startswith("L"):
           assert npz[k].shape == (len(npz["item_ids"]), 3584)
           assert npz[k].dtype == np.float16
   ```
3. **Accuracy sanity check (TASKS P1.3).** Report relaxed accuracy overall and
   separately for `human` and `augmented`. Qwen2.5-VL-7B's published ChartQA
   number is about **87 percent**. Landing far below that means the prompt or
   the metric is wrong, not the model. In particular, accuracy under about 60
   percent almost certainly means the answers are verbose prose and Section 4.1
   was not applied. **Stop and report rather than handing on a bad error set**,
   because every wrongly-marked item becomes a fake failure that someone then
   has to label by hand.
4. **Error yield.** Report how many rows have `correct == false`. Expect roughly
   300 to 500. This number decides whether the ChartQA train split has to be run
   as well, so report it as soon as it is known rather than at the end.
5. **No empty vision masks.** Assert during the run that every item had at least
   one vision token, and report the observed min and max token counts. They
   should sit inside 180 to 630.
6. **Staged back.** The `.npz` exists under `$HOME`, not only on a node's
   `/scratch`.

---

## 9. What not to do

- Do not change the pinned library versions, the dtype, the attention
  implementation, `max_pixels`, or `use_fast`. Each of them changes activations
  without changing much else, which is the hardest class of bug to find later.
- Do not filter to errors before caching. The binary probe needs the correct
  items too.
- Do not normalise, lowercase or otherwise clean the prediction strings before
  scoring. `is_correct` handles the numeric cases, and cleaning changes which
  items land in the error set.
- Do not tune the prompt beyond the single-word instruction in Section 4.1. If
  accuracy is far from the published number, report it rather than iterating on
  the prompt; a prompt tuned against the test split invalidates the numbers.
- Do not write activations only to `/scratch`.

---

## 10. What happens next, so you can see why the contract matters

Once the predictions file exists, the incorrect rows are labelled `structural`
or `fabrication` (or `ambiguous`) against the rubric in `docs/taxonomy.md`. Those
labels are written to `results/labels/qwen2_5_vl_7b_test.claude.jsonl`, keyed by
the **same `item_id`**.

The probe step then does roughly this, which is the whole reason Section 3 is
written the way it is:

```python
npz    = np.load("~/activations/qwen2_5_vl_7b/test.npz", allow_pickle=True)
order  = {iid: i for i, iid in enumerate(npz["item_ids"])}

rows   = read_predictions(...)          # keyed by item_id
labels = read_labels(...)               # keyed by item_id

idx = [order[r["item_id"]] for r in rows]            # rows -> activation rows
X   = {L: npz[f"L{L}_query_last"][idx] for L in layers}
y   = np.array([not r["correct"] for r in rows])     # binary probe
```

If the ids do not join, none of this runs. If they join by position instead of
by key, it runs and quietly produces a meaningless result, which is worse.
