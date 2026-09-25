# Qwen2.5-VL-7B ChartQA Output + Activation Caching Specification

## Purpose

Implement the **missing inference / activation-caching stage** for the existing repository.

The code must run `Qwen/Qwen2.5-VL-7B-Instruct` on the ChartQA **test split**, generate one deterministic answer per question, evaluate it with the repository's existing ChartQA relaxed-accuracy implementation, and cache the model's **pre-generation decoder activations from the same `generate()` call that produced that answer**.

The resulting prediction file and activation cache will later be joined by `item_id` to:

1. label incorrect outputs as structural / fabrication / ambiguous;
2. train a binary correct-vs-incorrect probe;
3. train failure-type probes;
4. sweep probe performance across decoder layers and pooling positions.

This file is an implementation contract for Codex. Follow the fixed decisions below exactly. If the repository disagrees with this document in a way that cannot be reconciled from existing code/configs, **fail loudly and report the conflict instead of guessing**.

---

# 1. Current experiment scope

This implementation is for the current Qwen extraction run only.

| Item | Fixed value |
| --- | --- |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Repository model key | `qwen2_5_vl_7b` |
| Dataset | ChartQA test only |
| Test composition | `test_human + test_augmented` |
| Expected questions | 2,500 |
| Distinct figures | 1,509 |
| Output predictions | all 2,500 questions |
| Output activations | all 2,500 questions |
| Precision | `torch.float16` |
| Attention implementation | `sdpa` |
| Quantization | forbidden |
| Image processor | `use_fast=True`, `max_pixels=1_000_000` |
| Decoder layers | all 28 unless config explicitly supplies a list |
| Hidden size | 3584 |
| Batch size | 1 |
| Decoding | greedy, deterministic |
| Max generation length | 32 new tokens |
| Activation time | pre-generation / prefill state |
| Activation source | same `generate()` invocation as the answer |
| Activation dtype on disk | `float16` |
| Final activation format | one compressed `.npz` |
| Primary durable storage | `$HOME` |
| Temporary run storage | node-local `/scratch` |

Do **not** implement LLaVA-NeXT in this task.

Do **not** expand to the ChartQA training split in this task. The test-run error yield determines whether that becomes necessary later.

Do **not** add visual-encoder-only features unless an existing committed config or downstream interface already requires them. The current activation contract is the four decoder representations defined in Section 8.

---

# 2. First inspect the existing repository

Before changing code, inspect and reuse the repository's existing implementations instead of re-implementing them.

Expected existing pieces from the current project:

```text
scripts/setup_env.sh
scripts/download_chartqa.sh
scripts/ada_env.sh

src/eval/chartqa.py
src/eval/relaxed_accuracy.py
src/eval/manifest.py

src/extract/smoke_qwen.py

src/probes/sweep.py

configs/inference.yaml
configs/activations.yaml
configs/model_configs/
```

The implementation should integrate with these files.

In particular:

- use `src.eval.chartqa.load_chartqa`;
- use `src.eval.relaxed_accuracy.is_correct`;
- extend/reuse the manifest schema rather than replacing it;
- reuse model/device/input preparation logic from `src/extract/smoke_qwen.py`;
- read activation layer selection from `configs/activations.yaml`;
- read inference settings from `configs/inference.yaml`;
- do not create duplicate loaders, duplicate scoring functions, or a parallel config system.

If one of these expected paths does not exist, inspect the repository for the equivalent existing implementation before creating anything new.

Prefer small functional helpers over a large class hierarchy.

---

# 3. Pinned runtime

The repository environment is already pinned. Do not upgrade packages and do not create a different environment.

Expected environment:

```bash
source $HOME/envs/vlmfail/bin/activate
```

Pinned versions:

```text
Python               3.10.12
torch                2.8.0+cu126
transformers         4.57.6
qwen-vl-utils        0.0.14
accelerate           1.14.0
```

A different `transformers` version is not an innocuous change: internal module paths, preprocessing, and activations may differ.

If the environment must be rebuilt, use the repository's existing setup script from a compute node. Do not build the environment on the Ada head node.

---

# 4. Ada hardware assumptions

Target hardware:

```text
RTX 2080 Ti
11 GiB VRAM per GPU
2 GPUs requested for the Qwen run
```

Qwen2.5-VL-7B does not fit on one 11 GiB card in FP16. Load it sharded across two GPUs using the existing smoke-test logic and:

```python
device_map="auto"
dtype=torch.float16
attn_implementation="sdpa"
```

Required model assertion:

```python
assert next(model.parameters()).dtype == torch.float16
```

Do not:

- quantize to 8-bit or 4-bit;
- use BF16;
- use FlashAttention-2;
- silently fall back to FP32.

The experiment measures hidden activations, so changing numerical implementation changes the object being measured.

---

# 5. Dataset loading

Use the repository loader:

```python
from src.eval.chartqa import load_chartqa

data = load_chartqa(os.path.expanduser("~/data/ChartQA"))
items = data["test_human"] + data["test_augmented"]
```

Expected item fields:

```text
figure_id
question
gold
split
source
```

Expected count:

```python
assert len(items) == 2500
```

The two sub-splits are **not figure-disjoint**. There are 103 figures shared between `test_human` and `test_augmented`, so downstream probe splitting must be performed by `figure_id`, not by row or by the human/augmented source split.

Always preserve `figure_id` in both prediction and activation artefacts.

Do not parse ChartQA JSON manually if the existing loader works.

---

# 6. Stable item identity

Every artefact must join by a deterministic `item_id`.

**Do not define this function. Import the one already committed:**

```python
from src.label.pool import item_id

iid = item_id("qwen2_5_vl_7b", figure_id, question, source)
```

For reference, that committed function is:

```python
def item_id(model: str, figure_id: str, question: str, source: str) -> str:
    raw = f"{model}\x1f{source}\x1f{figure_id}\x1f{question}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]
```

> **Corrected 25 September.** An earlier version of this section specified a
> local `make_item_id` using `|` as the separator and 16 hex characters. That
> does not match `src/label/pool.py`, which the labelling pipeline already uses,
> and the mismatch does not surface as an error. Worked example:
>
> ```text
> this spec's old version : ef18c807279b8cf3
> src/label/pool.py       : fba76b41110d
> ```
>
> The reason it is silent is the part worth understanding.
> `src/eval/manifest.py::read_manifest` reads only its five named fields and
> **drops `item_id` entirely**. `src/label/pool.py::build_pool` then recomputes
> the id with its own function. So predictions and activations would be keyed
> one way, the annotation pool and every label keyed another, and the two sets
> would never intersect. The binary probe would still train, because it joins
> predictions to activations and both use the same wrong key. The failure-type
> probes would come out with **zero items**, discovered only after the seven
> hour extraction run and the entire labelling effort.
>
> `\x1f` is the ASCII unit separator and cannot occur inside a question or a
> filename, which is why the committed version is also the better one.

Call it with the literal model key:

```python
model = "qwen2_5_vl_7b"
```

Not the Hugging Face id, not a display name, and not the shorter `qwen`. The
model string is hashed, so any variation changes every id.

Important:

- `figure_id` is used exactly as supplied by the loader;
- `question` is the raw, unmodified question;
- do not strip, lowercase, normalize, or otherwise rewrite either field before hashing;
- compute `item_id` once and use the same value for predictions, activation shards, final activation cache, resumability, and downstream labels.

Never use file position or dataset index as the join key.

---

# 7. Prompt and generation

## 7.1 Exact prompt suffix

Use:

```python
SUFFIX = "\nAnswer the question using a single word or phrase."
```

Build the Qwen message in this form:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question + SUFFIX},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

The purpose of the suffix is to keep answers compatible with ChartQA's short-answer evaluation.

Do not prompt-tune against the test split.

## 7.2 Processor

Use the pinned processor behavior:

```python
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    max_pixels=1_000_000,
    use_fast=True,
)
```

Do not change `max_pixels` or `use_fast`.

Expected visual-token count for current ChartQA figures is approximately 180-630. Assert that a visual mask exists for every sample and record observed min/max visual-token counts.

## 7.3 Deterministic generation

Use project seed 42:

```python
torch.manual_seed(42)
```

Generate with:

```python
do_sample=False
max_new_tokens=32
```

No temperature, top-p, top-k sampling, beam search, or stochastic decoding.

Decode only the generated continuation, not the prompt tokens.

After decoding:

```python
prediction = prediction.strip()
```

Do not lowercase it, strip punctuation, remove units, rewrite numbers, or otherwise normalize it before scoring.

---

# 8. Activation contract

## 8.1 Scientific requirement

The cached state must be the model state:

> after reading the image and question, but before emitting the first answer token.

That is the **prefill forward pass** inside `model.generate()`.

The activation and answer must come from the **same `generate()` invocation**.

Forbidden:

```text
generate answer
then run a separate forward pass for activations
```

Also forbidden:

```python
generate(..., output_hidden_states=True)
```

The latter retains hidden states across autoregressive decoding steps and wastes memory.

## 8.2 Capture method

Register forward hooks on the selected language decoder layers before calling `generate()`.

For each item:

1. clear the capture state;
2. enable capture;
3. call `model.generate(...)`;
4. each layer hook records **only its first invocation** for that item;
5. subsequent decode-step invocations from the same `generate()` call are ignored;
6. verify exactly one prefill capture was obtained from every requested layer;
7. disable/reset capture before processing the next item.

The first decoder invocation during generation is the prefill pass.

Do not rely on global hook state that can accidentally leak from one example into the next.

## 8.3 Find and verify decoder layers

With the pinned Transformers version, the expected path is approximately:

```python
layers = model.model.language_model.layers
```

However, treat this as something to verify, not blindly assume.

The loaded decoder-layer collection must satisfy:

```python
assert len(layers) == 28
```

The hidden state emitted by every hooked decoder block must satisfy:

```python
h.ndim == 3
h.shape[0] == 1
h.shape[-1] == 3584
```

Layer index `i` in the output means:

> output of decoder block `i`, using zero-based indexing.

Do not silently shift layer numbering by one.

## 8.4 Layer selection

Read `configs/activations.yaml`.

Support:

```yaml
layers: all
```

and an explicit integer list.

If `layers: all`, resolve to:

```python
list(range(28))
```

For the current experiment, cache all 28 layers unless the committed config explicitly says otherwise.

Record the fully resolved list in run metadata.

Do not hardcode a five-layer subset.

## 8.5 Four pooled representations per layer

For each requested decoder layer, cache exactly these four vectors:

| Position key | Definition |
| --- | --- |
| `vision_mean` | elementwise mean of hidden states at image-token positions |
| `vision_max` | elementwise max of hidden states at image-token positions |
| `query_last` | hidden state at the final valid prompt token |
| `query_mean` | elementwise mean over valid non-image, non-special text-token positions |

Each resulting vector has shape:

```text
(3584,)
```

Each final layer array has shape:

```text
(N, 3584)
```

where `N == 2500`.

### Vision mask

Construct from prompt `input_ids`:

```python
vision_mask = input_ids == image_token_id
```

Resolve `image_token_id` from the loaded model/config rather than inventing a numeric constant.

Assert:

```python
vision_mask.any()
```

and record its token count for diagnostics.

The hidden-state sequence length must align with the mask length. Assert it. If it does not, stop and inspect the loaded Qwen implementation rather than attempting an index workaround.

### `query_last`

Use the last valid prompt position according to `attention_mask`, not a generated token.

At batch size 1 with no padding this will normally be the final sequence position, but compute it from the mask so the definition is explicit.

### `query_mean`

Construct a mask over valid prompt text tokens:

```text
attention_mask == 1
AND not an image token
AND not a tokenizer special token
```

Use the tokenizer's loaded special-token IDs; do not hardcode them.

Assert the text mask is non-empty.

### Pooling and transfer

Pool only the required vectors inside the hook/capture path. Do not retain the whole `(1, T, 3584)` tensor after the item.

Detach the pooled vectors, move them to CPU, and store them as `float16`.

The GPU cache should not grow with dataset size.

---

# 9. Why all layers are cached

The failure-type hypothesis is explicitly depth-dependent. Structural misreading and fabrication may become separable at different parts of the network.

Dropping layers before observing the layer-wise probe curve risks missing the useful depth and would require rerunning the expensive VLM inference.

Storage is small enough to avoid that:

```text
28 layers
x 4 pooled vectors/layer
x 3584 values/vector
x 2 bytes/value
= 802,816 bytes/item
≈ 784 KiB/item
```

For 2,500 examples:

```text
≈ 1.9 GiB
```

Therefore cache all layers now and select layers later during probe analysis.

---

# 10. Prediction output contract

Write:

```text
results/predictions/qwen2_5_vl_7b_test.jsonl
```

One JSON object per question.

Required schema:

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

Required semantics:

- `item_id`: stable SHA-1-derived join key defined above;
- `model`: exactly `qwen2_5_vl_7b`;
- `figure_id`: exactly loader value;
- `question`: raw loader question;
- `gold`: raw gold answer used by scorer;
- `prediction`: stripped decoded model answer;
- `correct`: result of repository relaxed accuracy;
- `split`: expected `test`;
- `source`: `human` or `augmented`.

Use the repository scorer:

```python
from src.eval.relaxed_accuracy import is_correct

correct = is_correct(gold, prediction)
```

Do not reimplement relaxed accuracy.

---

# 11. Activation output contract

Final durable file:

```text
$HOME/activations/qwen2_5_vl_7b/test.npz
```

The `.npz` must contain:

```text
item_ids
figure_ids
L0_vision_mean
L0_vision_max
L0_query_last
L0_query_mean
L1_vision_mean
...
L27_query_mean
```

when all layers are enabled.

Conceptual save:

```python
np.savez_compressed(
    path,
    item_ids=np.asarray(item_ids, dtype="<U16"),
    figure_ids=np.asarray(figure_ids, dtype=object),
    **activation_arrays,
)
```

For each activation key:

```python
activation_arrays[f"L{layer}_{position}"].shape == (N, 3584)
activation_arrays[f"L{layer}_{position}"].dtype == np.float16
```

Row alignment rule:

```text
item_ids[i]
figure_ids[i]
every L*_*.npy row i
```

must all describe the same example.

The `item_ids` array is authoritative for joining activation rows back to predictions and labels.

Do not create one final activation file per question.

---

# 12. Probe compatibility

The final cache must support downstream usage equivalent to:

```python
npz = np.load(
    os.path.expanduser("~/activations/qwen2_5_vl_7b/test.npz"),
    allow_pickle=True,
)

order = {iid: i for i, iid in enumerate(npz["item_ids"])}

rows = read_predictions(...)
labels = read_labels(...)

idx = [order[r["item_id"]] for r in rows]

X = {
    layer: npz[f"L{layer}_query_last"][idx]
    for layer in layers
}

y = np.asarray([not r["correct"] for r in rows])
```

For failure-type probes, incorrect prediction rows are joined to the later label file by the same `item_id`, then the corresponding activation rows are selected from this cache.

The cache must therefore contain **correct and incorrect examples**.

Never cache only failures.

---

# 13. Resumability

The extraction job must survive preemption / walltime loss.

## 13.1 Prediction rows

Append one prediction JSON object immediately after each successfully completed item.

After writing each row:

```python
f.flush()
os.fsync(f.fileno())
```

On restart:

1. read the existing JSONL;
2. build a set of completed `item_id`s;
3. skip those examples.

The item ID, not row number, is the resumability key.

## 13.2 Activation shards

A final `.npz` is not appendable, so do not attempt to rewrite the full `.npz` after every item.

During extraction, write resumable activation shards to node-local scratch.

Recommended shard size:

```text
250 items
```

A shard must carry:

```text
item_ids
figure_ids
all configured layer/position arrays
```

The shard format may be `.npz` or a small set of `.npy` files, provided:

- it is written atomically;
- it carries its own `item_ids`;
- it can be reloaded independently;
- final assembly does not rely on shard filename order.

If writing a shard to a final filename, write to a temporary filename first and rename only after the shard is complete.

## 13.3 Resume consistency

A prediction row and its activations form one logical completed item.

Do not consider an item resumably complete merely because a prediction row exists if the corresponding activation row is absent from all persisted shards.

On startup, derive the set of safely completed examples from the intersection of durable prediction IDs and durable activation-shard IDs, or otherwise implement an equivalent mechanism that cannot skip missing activations.

If the current repository already defines a resumability convention, use it, but preserve this invariant.

## 13.4 Final assembly

After all 2,500 examples have both a prediction row and activation data:

1. load all shards;
2. reject duplicate `item_id`s;
3. assemble a deterministic row order;
4. concatenate each activation key using exactly that order;
5. write the final compressed `.npz`;
6. assert final IDs exactly match prediction IDs.

Required invariant:

```python
set(npz["item_ids"]) == {row["item_id"] for row in predictions}
```

and:

```python
len(npz["item_ids"]) == len(set(npz["item_ids"]))
```

---

# 14. Scratch and durable storage

Ada `/scratch` is node-local and temporary.

Use it for:

- temporary activation shards;
- temporary assembly files;
- other high-write intermediate data.

Before successful job exit, stage the completed activation cache back to:

```text
$HOME/activations/qwen2_5_vl_7b/test.npz
```

Use the repository's existing `scripts/ada_env.sh` / `stage_out` mechanism where applicable.

Do not leave the only copy under `/scratch`.

The prediction JSONL must also be durable.

---

# 15. Correctness checks during extraction

The implementation must fail early rather than silently generate unusable activations.

For every run, validate:

### Model

```python
assert next(model.parameters()).dtype == torch.float16
assert len(layers) == 28
```

### Hook output

For every selected layer on every prefill:

```python
assert h.ndim == 3
assert h.shape[0] == 1
assert h.shape[-1] == 3584
```

### Masks

For every item:

```python
assert vision_mask.any()
assert text_mask.any()
assert hidden_sequence_length == mask_sequence_length
```

Record visual-token count.

### Capture count

For every item:

```text
every requested layer captured exactly once as the prefill representation
no requested layer missing
```

### Numerical sanity

Reject or count/report any activation containing NaN or Inf.

At minimum:

```python
assert torch.isfinite(pooled_vector).all()
```

before persisting.

### Output shape

Each final activation array:

```python
shape == (2500, 3584)
dtype == np.float16
```

---

# 16. Final acceptance criteria

The job is complete only when all checks below pass.

## A. Predictions

```text
results/predictions/qwen2_5_vl_7b_test.jsonl
```

must have:

- exactly 2,500 rows;
- exactly 2,500 unique `item_id`s;
- required fields present;
- no duplicated `(model, figure_id, question)` identity.

## B. Activations

```text
$HOME/activations/qwen2_5_vl_7b/test.npz
```

must contain:

- 2,500 `item_ids`;
- 2,500 `figure_ids`;
- one `(2500, 3584)` float16 array for every requested `L{layer}_{position}` key;
- no duplicate item IDs;
- same item-ID set as predictions.

For `layers: all`, expected number of activation matrices:

```text
28 layers x 4 positions = 112 matrices
```

plus metadata arrays.

## C. Accuracy sanity report

Report relaxed accuracy:

```text
overall
human
augmented
```

Published Qwen2.5-VL-7B ChartQA performance is around the high-80% range. The exact number in this run can differ, but a severe collapse is a pipeline bug until investigated.

If overall relaxed accuracy is below roughly 60%, stop before handing the outputs downstream and inspect:

- prompt construction;
- whether only generated continuation was decoded;
- image preprocessing;
- model loading / dtype;
- scorer usage.

Do not "fix" poor test accuracy by iteratively prompt-tuning on the test split.

## D. Error yield

Report:

```text
n_correct
n_incorrect
error_rate
```

Expected incorrect count is roughly a few hundred.

This determines whether the project later needs to run more ChartQA examples.

## E. Vision-token statistics

Report:

```text
minimum visual-token count
maximum visual-token count
mean / median if convenient
```

There must be no empty vision mask.

## F. Durable stage-out

The final activation cache must exist under `$HOME`, not only on `/scratch`.

---

# 17. Required run metadata

Write a small machine-readable run metadata file next to the outputs, using the repository's existing convention if one exists.

It should record at least:

```text
model Hugging Face ID
repository model key
resolved model revision if available
Python version
torch version
transformers version
qwen-vl-utils version
accelerate version
dtype
attention implementation
device map
GPU names
number of GPUs
processor use_fast value
max_pixels
prompt suffix
do_sample
max_new_tokens
seed
requested layer config
resolved layer list
hidden size
pooling position names
dataset path
dataset split composition
number of items
start/end timestamps if the repository already records them
```

This metadata is for reproducibility. Do not invent a second experiment configuration; serialize the resolved values actually used.

---

# 18. Suggested implementation decomposition

Keep implementation small and functional. The exact filenames should follow repository conventions.

A minimal decomposition is:

```python
# item_id is imported from src.label.pool, not written here
load_model_and_processor(...)
resolve_decoder_layers(...)
resolve_layer_indices(...)
build_prompt_and_inputs(...)
build_pool_masks(...)
register_prefill_hooks(...)
generate_one(...)
score_one(...)
append_prediction(...)
write_activation_shard(...)
load_completed_ids(...)
assemble_final_npz(...)
validate_final_outputs(...)
```

Avoid unnecessary classes unless the repository already uses one for extraction state.

The hook implementation may use a small mutable capture object/dict local to the extraction module if needed, but it must be reset explicitly per sample.

---

# 19. Per-item algorithm

The extraction loop should be logically equivalent to:

```python
for item in items:
    iid = item_id(                       # from src.label.pool
        "qwen2_5_vl_7b",
        item.figure_id,
        item.question,
    )

    if item_is_safely_completed(iid):
        continue

    image = load_image_using_existing_repo_logic(item.figure_id)

    prompt, inputs = build_prompt_and_inputs(
        processor=processor,
        image=image,
        question=item.question,
    )

    masks = build_pool_masks(
        inputs=inputs,
        tokenizer=processor.tokenizer,
        image_token_id=resolved_image_token_id,
    )

    reset_capture_state(iid, masks)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=32,
        )

    assert_all_requested_prefill_layers_captured()

    prediction = decode_generated_continuation(
        processor,
        inputs,
        generated,
    ).strip()

    correct = is_correct(item.gold, prediction)

    persist_prediction_row(...)

    add_pooled_activations_to_current_shard(
        item_id=iid,
        figure_id=item.figure_id,
        ...
    )

    if shard_is_full:
        persist_shard_atomically(...)
```

At the end:

```python
flush_partial_shard()
assemble_final_npz()
validate_final_outputs()
stage_out()
write_summary_report()
```

This is pseudocode only. Reuse existing repository helpers for input/image/device handling.

---

# 20. Important generation/hook detail

`generate()` invokes the language model repeatedly:

```text
prefill over the whole prompt
decode token 1
decode token 2
...
```

Therefore a decoder hook will fire multiple times.

For this project, only the first invocation per layer per sample is valid.

A safe logical pattern is:

```python
captured = {}

def make_hook(layer_idx):
    def hook(module, args, output):
        if layer_idx in captured:
            return

        h = output[0] if isinstance(output, tuple) else output
        validate_hidden(h)

        captured[layer_idx] = pool_required_vectors(
            h,
            masks=current_masks,
        )
    return hook
```

The real code must account for the exact output type returned by the pinned Qwen decoder block. Inspect the smoke test / loaded model and handle the actual type explicitly.

Do not write a generic "try every possible tuple/object format" compatibility layer. The environment is pinned, so explicit validated handling is preferable.

---

# 21. Final cache key convention

The exact key template is:

```text
L{zero_based_layer_index}_{position}
```

Allowed `position` values:

```text
vision_mean
vision_max
query_last
query_mean
```

Examples:

```text
L0_query_last
L7_vision_mean
L14_vision_max
L27_query_mean
```

Do not rename these to abbreviations such as `qt`, `vt`, `qmean`, etc.

Downstream code should not need a translation layer.

---

# 22. Probe split warning

The extraction code itself does not train probes, but it must preserve enough metadata to make probe evaluation valid.

The probe train/validation/test split must later be **figure-level**, because multiple questions can reference the same chart.

Do not split activation rows independently by question.

Keeping `figure_ids` parallel to `item_ids` in the final `.npz` is mandatory for this reason.

---

# 23. Things Codex must not do

Do not:

- implement LLaVA in this task;
- run the ChartQA train split;
- cache only incorrect examples;
- use a separate activation forward pass;
- use `output_hidden_states=True` during `generate()`;
- cache full token-by-layer hidden-state tensors;
- quantize the model;
- use BF16;
- use FlashAttention-2;
- change pinned dependency versions;
- change processor mode;
- change `max_pixels`;
- prompt-tune on ChartQA test;
- reimplement relaxed accuracy;
- normalize prediction strings beyond `.strip()`;
- use row number as an identifier;
- define a local `item_id` / `make_item_id` instead of importing
  `src.label.pool.item_id` (see Section 6: a near-miss here does not raise,
  it silently empties the failure-type probes);
- assume human/augmented are figure-disjoint;
- hardcode an image-token numeric ID;
- hardcode tokenizer special IDs;
- hardcode decoder layers without validating the loaded model;
- silently skip an unexpected tensor shape;
- silently tolerate missing hook captures;
- leave the only activation copy on `/scratch`;
- create 2,500 final per-item activation files;
- write final float32 activations;
- add new dependencies unless absolutely required.

If an invariant fails, stop with a useful error.

---

# 24. Deliverables from the coding task

The coding task is finished when the repository contains the minimal implementation necessary to perform this run and document how to launch it.

Expected deliverables:

1. extraction implementation integrated into the existing repository;
2. any minimal config changes required to expose the already-decided settings;
3. resumable activation shard handling;
4. final `.npz` assembly/validation;
5. prediction JSONL writing;
6. run metadata / summary report;
7. an Ada launch command or Slurm script following the repository's existing job-script style;
8. no unrelated refactors.

Before the full 2,500-item run, execute the repository's existing one-item smoke path or a tiny extraction subset only to verify:

```text
model loads on 2 GPUs
image path resolves
prompt works
generate() succeeds
28 hooks are found
all requested layers capture the prefill
four pooled vectors have length 3584
vectors are finite
prediction scoring works
shard writing works
resume logic works
```

This is a correctness check, not a separate experiment.

---

# 25. Expected final artefacts

After the full run:

```text
results/
└── predictions/
    └── qwen2_5_vl_7b_test.jsonl

$HOME/activations/
└── qwen2_5_vl_7b/
    └── test.npz
```

Plus the repository's normal run metadata / summary output.

The prediction and activation artefacts must join exactly by `item_id`.

Later labeling will create an additional label file keyed by the same ID; the activation extractor does not need to implement that stage.

---

# 26. Downstream mental model

The complete data flow is:

```text
ChartQA item
    |
    | image + question
    v
Qwen2.5-VL-7B generate()
    |
    |-- first decoder invocation = prefill
    |      |
    |      `-- cache four pooled vectors at every requested decoder layer
    |
    `-- greedy generated answer
             |
             `-- ChartQA relaxed scoring
                      |
                      `-- prediction JSONL row

prediction.item_id
        ==
activation.item_ids[row]
        ==
later failure-label.item_id
```

That equality is the core interface of the project.

---

# 27. Final implementation principle

The expensive part of the project is the frozen VLM forward/generation run. Probe training later is cheap.

Therefore prefer:

```text
cache enough validated information now
```

over:

```text
save storage by discarding layers/positions and rerun the VLM later
```

For the current 2,500-item Qwen run, caching all 28 decoder layers at the four required pooled positions is only about 1.9 GiB and should be treated as the default.
