"""Annotation and agreement UI (TASKS P2.3 to P2.6). Run from the repo root:

    streamlit run src/label/app.py

Two pages. **Annotate** walks your task file one item at a time and appends each
label to `annotations/<you>.jsonl` the moment you click, so closing the tab
loses nothing and reopening resumes where you stopped. **Agreement** reads every
rater file in `annotations/` and reports kappa against the 0.6 gate.

Blind by design: you never see which model produced an answer (pool.py hashes
the model into an opaque ID and shuffles it out of order), and you never see a
teammate's label for the same item. Do not open another person's file or
`annotations/tasks/pool_key.jsonl` until you have finished yours.
"""

from __future__ import annotations

import json
import os
import pathlib

import streamlit as st

from src.label.annotation import (
    AnnotationItem, append_annotation, load_completed_ids, load_items, make_record,
)
from src.label.multi_rater import (
    KAPPA_THRESHOLD, MIN_PAIR_ITEMS, disagreements, fleiss_kappa, label_counts,
    load_raters, pairwise, shared_items,
)
from src.label.pool import sanitize

REPO = pathlib.Path(__file__).resolve().parents[2]
ANN_DIR = REPO / "annotations"
TASK_DIR = ANN_DIR / "tasks"
TAXONOMY = REPO / "docs" / "taxonomy.md"
LABELS = [("Structural", "structural"), ("Fabrication", "fabrication"),
          ("Ambiguous", "ambiguous")]

st.set_page_config(page_title="Failure-type annotation", layout="wide")


@st.cache_data
def _items(path: str, image_root: str, mtime: float) -> list[AnnotationItem]:
    # mtime is only a cache key, so an edited task file is picked up on reload
    return load_items(path, image_base_dir=image_root)


def _save(item: AnnotationItem, label: str, name: str, out: pathlib.Path) -> None:
    rationale = st.session_state.get(f"why_{item.item_id}", "")
    append_annotation(make_record(item, label, rationale, name), out)


def _last_record(out: pathlib.Path) -> dict | None:
    if not out.is_file():
        return None
    lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def _relabel(rec: dict, label: str, name: str, out: pathlib.Path) -> None:
    item = AnnotationItem(rec["item_id"], "", rec["question"],
                          rec["gold_answer"], rec["model_answer"])
    append_annotation(make_record(item, label, rec.get("rationale", ""), name), out)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")
    name = st.text_input("Your name", help="Must match your task file, e.g. naman")
    image_root = st.text_input(
        "ChartQA folder", os.environ.get("CHARTQA_ROOT", str(REPO / "data" / "ChartQA")),
        help="The folder scripts/download_chartqa.sh created, containing test/png/")
    stem = sanitize(name) if name.strip() else ""
    task_file = st.text_input(
        "Task file", str(TASK_DIR / f"{stem}.jsonl") if stem else "",
        help="Written by `python -m src.label.pool assign`")
    page = st.radio("Page", ["Annotate", "Agreement"])


# ---------------------------------------------------------------------------
# Annotate
# ---------------------------------------------------------------------------

def annotate() -> None:
    if not stem:
        st.info("Enter your name in the sidebar to start.")
        return
    tasks = pathlib.Path(task_file)
    if not tasks.is_file():
        st.error(f"No task file at `{tasks}`. Ask whoever ran `pool assign` for yours.")
        return

    out = ANN_DIR / f"{stem}.jsonl"
    items = _items(str(tasks), image_root, tasks.stat().st_mtime)
    done = load_completed_ids(out)
    remaining = [it for it in items if it.item_id not in done]
    n_done = len(items) - len(remaining)

    st.progress(n_done / len(items) if items else 1.0,
                text=f"{n_done} of {len(items)} labelled, saving to `{out.relative_to(REPO)}`")

    with st.expander("Rubric (docs/taxonomy.md)"):
        st.markdown(TAXONOMY.read_text(encoding="utf-8") if TAXONOMY.is_file()
                    else "docs/taxonomy.md not found.")

    if not remaining:
        st.success("All done. Commit and push `annotations/"
                   f"{stem}.jsonl`, then check the Agreement page.")
    else:
        item = remaining[0]
        left, right = st.columns([3, 2])
        with left:
            if pathlib.Path(item.image_path).is_file():
                st.image(item.image_path, width="stretch")
            else:
                st.warning(f"Image not found: `{item.image_path}`. "
                           "Check the ChartQA folder in the sidebar.")
        with right:
            st.caption(f"item {item.item_id}")
            st.markdown(f"**Question**\n\n{item.question}")
            st.markdown(f"**Gold answer**\n\n{item.gold_answer}")
            st.markdown(f"**Model answer**\n\n{item.model_answer}")
            st.text_input("One line on why (optional, but helps adjudication)",
                          key=f"why_{item.item_id}")
            cols = st.columns(3)
            for col, (text, label) in zip(cols, LABELS):
                col.button(text, key=f"{label}_{item.item_id}", width="stretch",
                           on_click=_save, args=(item, label, stem, out))

    last = _last_record(out)
    if last:
        st.divider()
        st.caption(f"Last saved: **{last['label']}** for \"{last['question']}\". "
                   "Misclicked? Change it here; the newer label replaces the old one.")
        cols = st.columns(3)
        for col, (text, label) in zip(cols, LABELS):
            col.button(f"Change to {text}", key=f"re_{label}", disabled=label == last["label"],
                       on_click=_relabel, args=(last, label, stem, out))


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def agreement() -> None:
    ratings, bad = load_raters(ANN_DIR)
    if bad:
        st.warning(f"Skipped {bad} malformed line(s) across the rater files.")
    if len(ratings) < 2:
        st.info(f"Need at least two rater files in `annotations/`; found {len(ratings)}.")
        return

    llm = [r for r in ratings if "claude" in r or "llm" in r]
    selected = st.multiselect(
        "Raters in the headline kappa", sorted(ratings),
        default=[r for r in sorted(ratings) if r not in llm],
        help="LLM labellers are excluded by default: the 0.6 gate is about whether "
             "humans can apply the rubric consistently. Their agreement with humans "
             "is in the pairwise table below.")

    shared = shared_items(ratings, selected)
    kappa = fleiss_kappa(list(shared.values()))
    c1, c2, c3 = st.columns(3)
    c1.metric("Fleiss kappa", "n/a" if kappa != kappa else f"{kappa:.3f}")
    c2.metric("Items with 2+ labels", len(shared))
    c3.metric("Gate", f"> {KAPPA_THRESHOLD}")
    if kappa != kappa:
        st.info("Not enough overlapping labels yet to compute kappa.")
    elif kappa > KAPPA_THRESHOLD:
        st.success("Passes the gate.")
    else:
        st.error("Below the gate. Adjudicate the disagreements below and revise "
                 "docs/taxonomy.md before labelling further (TASKS P2.4).")

    st.subheader("Pairwise Cohen's kappa")
    st.caption(f"Pairs sharing fewer than {MIN_PAIR_ITEMS} items are noise; read them "
               "for direction only. One annotator low against everyone else means "
               "their reading of the rubric has drifted.")
    st.dataframe(pairwise(ratings), width="stretch")

    st.subheader("Label distribution")
    st.dataframe([{"rater": r, **c} for r, c in label_counts(ratings).items()],
                 width="stretch")

    rows = disagreements(ratings, selected)
    st.subheader(f"Disagreements ({len(rows)})")
    pool_file = TASK_DIR / "pool.jsonl"
    images = {}
    if pool_file.is_file():
        for line in pool_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                images[p["item_id"]] = pathlib.Path(image_root) / p["image_path"]
    for d in rows:
        votes = ", ".join(f"{r}: {v[0]}" for r, v in d["votes"].items())
        with st.expander(f"{d['question']}  ({votes})"):
            img = images.get(d["item_id"])
            if img is not None and img.is_file():
                st.image(str(img), width=520)
            st.markdown(f"**Gold** {d['gold_answer']}  \n**Model** {d['model_answer']}")
            for r, (label, why) in d["votes"].items():
                st.markdown(f"- **{r}**: {label}" + (f", \"{why}\"" if why else ""))
            st.caption(f"item {d['item_id']}")


if page == "Annotate":
    annotate()
else:
    agreement()
