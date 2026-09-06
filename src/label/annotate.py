"""Minimal blind annotation tool (TASKS P2.2).

Show the figure, question, gold answer and model's answer; take one keypress
for `structural`, `fabrication` or `ambiguous`; take a one-line rationale;
append a row to disk; move on.

Strictly satisfies:
- Blinding: annotators never see the model identity or each other's labels.
- Immediate disk persistence: appends and flushes every record after each item.
- Minimal overhead: CLI with single keypress or a compact Tkinter GUI.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Optional

from src.label.annotation import (
    ALLOWED_LABELS,
    KEY_MAP,
    AnnotationItem,
    append_annotation,
    load_completed_ids,
    load_items,
    make_record,
)


def _get_single_keypress() -> str:
    """Read a single character from the console without requiring Enter on Windows/Unix."""
    # Windows interactive console
    if sys.platform == "win32" and sys.stdin.isatty():
        try:
            import msvcrt
            while True:
                char = msvcrt.getch()
                if char in (b"\x03", b"\x1a"):  # Ctrl+C or Ctrl+Z
                    raise KeyboardInterrupt
                try:
                    decoded = char.decode("utf-8", errors="ignore").lower()
                except Exception:
                    continue
                if decoded in ("s", "f", "a", "q"):
                    return decoded
        except ImportError:
            pass

    # Fallback to standard input (works with pipes, automation, or non-Windows)
    sys.stdout.write("Enter choice [s/f/a/q]: ")
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        return "q"
    choice = line.strip().lower()
    return choice[0] if choice else "q"


def run_cli(
    items: list[AnnotationItem],
    output_path: pathlib.Path,
    annotator: str,
    limit: Optional[int] = None,
    open_image: bool = True,
) -> int:
    """Run interactive terminal-based annotation."""
    completed_ids = load_completed_ids(output_path)
    pending_items = [item for item in items if item.item_id not in completed_ids]

    if limit is not None and limit > 0:
        pending_items = pending_items[:limit]

    total_pending = len(pending_items)
    if total_pending == 0:
        print(f"All {len(items)} items in the input set are already annotated in {output_path}.")
        return 0

    print(f"\n--- Starting Annotation Session ---")
    print(f"Annotator: {annotator}")
    print(f"Pending items: {total_pending} (out of {len(items)} total)")
    print(f"Destination: {output_path}")
    print(f"Commands: [s] = Structural | [f] = Fabrication | [a] = Ambiguous | [q] = Quit\n")

    annotated_count = 0
    for idx, item in enumerate(pending_items, 1):
        print("=" * 80)
        print(f"[{idx}/{total_pending}] Item ID: {item.item_id}")
        print(f"Figure: {item.image_path}")
        print(f"Question:     {item.question}")
        print(f"Gold Answer:  {item.gold_answer}")
        print(f"Model Answer: {item.model_answer}")
        print("-" * 80)

        # Open figure image in external viewer if requested and file exists
        if open_image and item.image_path and pathlib.Path(item.image_path).is_file():
            try:
                if sys.platform == "win32":
                    os.startfile(item.image_path)  # type: ignore
                elif sys.platform == "darwin":
                    os.system(f"open '{item.image_path}'")
                else:
                    os.system(f"xdg-open '{item.image_path}' &")
            except Exception:
                pass

        sys.stdout.write("Select [s]tructural, [f]abrication, [a]mbiguous (or [q]uit): ")
        sys.stdout.flush()

        key = _get_single_keypress()
        print(key)

        if key == "q":
            print("\nSession paused by annotator. Progress saved.")
            break

        if key not in KEY_MAP:
            print(f"Invalid key {key!r}. Skipping to next prompt.")
            continue

        label = KEY_MAP[key]
        sys.stdout.write(f"Rationale for {label} (one line): ")
        sys.stdout.flush()
        rationale = sys.stdin.readline().strip()
        if not rationale:
            rationale = f"Observed {label} failure according to taxonomy."

        record = make_record(
            item=item,
            label=label,
            rationale=rationale,
            annotator=annotator,
        )
        append_annotation(record, output_path)
        annotated_count += 1
        print(f"[OK] Saved ({label}). Total annotated this session: {annotated_count}\n")

    print("=" * 80)
    print(f"Annotation session concluded. {annotated_count} items appended to {output_path}.")
    return annotated_count


def run_gui(
    items: list[AnnotationItem],
    output_path: pathlib.Path,
    annotator: str,
    limit: Optional[int] = None,
) -> int:
    """Run minimal Tkinter GUI for blind annotation."""
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk

    completed_ids = load_completed_ids(output_path)
    pending = [it for it in items if it.item_id not in completed_ids]
    if limit is not None and limit > 0:
        pending = pending[:limit]

    if not pending:
        print("No pending items for GUI annotation.")
        return 0

    root = tk.Tk()
    root.title(f"Blind Annotation - {annotator}")
    root.geometry("820x840")

    state = {"index": 0, "annotated": 0, "selected_label": tk.StringVar(value="")}

    # Top status bar
    status_label = tk.Label(root, font=("Segoe UI", 11, "bold"), fg="#333333", pady=6)
    status_label.pack(fill=tk.X)

    # Image canvas/frame
    img_label = tk.Label(root, bg="#f0f0f0")
    img_label.pack(padx=10, pady=5)

    # Info frame
    info_frame = tk.Frame(root, padx=15, pady=10)
    info_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(info_frame, text="Question:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="nw", pady=2)
    q_val = tk.Label(info_frame, wraplength=650, justify="left", font=("Segoe UI", 10))
    q_val.grid(row=0, column=1, sticky="w", pady=2)

    tk.Label(info_frame, text="Gold Answer:", font=("Segoe UI", 10, "bold"), fg="#1b5e20").grid(row=1, column=0, sticky="nw", pady=2)
    gold_val = tk.Label(info_frame, wraplength=650, justify="left", font=("Segoe UI", 10, "bold"), fg="#1b5e20")
    gold_val.grid(row=1, column=1, sticky="w", pady=2)

    tk.Label(info_frame, text="Model Answer:", font=("Segoe UI", 10, "bold"), fg="#b71c1c").grid(row=2, column=0, sticky="nw", pady=2)
    model_val = tk.Label(info_frame, wraplength=650, justify="left", font=("Segoe UI", 10, "bold"), fg="#b71c1c")
    model_val.grid(row=2, column=1, sticky="w", pady=2)

    tk.Label(info_frame, text="Rationale:", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="nw", pady=6)
    rationale_entry = tk.Entry(info_frame, font=("Segoe UI", 10), width=60)
    rationale_entry.grid(row=3, column=1, sticky="w", pady=6)

    # Action buttons
    btn_frame = tk.Frame(root, pady=10)
    btn_frame.pack()

    current_img_ref = [None]

    def display_current():
        idx = state["index"]
        if idx >= len(pending):
            messagebox.showinfo("Done", f"Finished annotating {state['annotated']} items!")
            root.destroy()
            return

        item = pending[idx]
        status_label.config(
            text=f"Item {idx + 1} of {len(pending)}  |  ID: {item.item_id}"
        )
        q_val.config(text=item.question)
        gold_val.config(text=item.gold_answer)
        model_val.config(text=item.model_answer)
        rationale_entry.delete(0, tk.END)
        state["selected_label"].set("")

        if item.image_path and pathlib.Path(item.image_path).is_file():
            try:
                pil_img = Image.open(item.image_path)
                pil_img.thumbnail((640, 420))
                tk_img = ImageTk.PhotoImage(pil_img)
                img_label.config(image=tk_img, text="")
                current_img_ref[0] = tk_img
            except Exception as e:
                img_label.config(image="", text=f"(Failed to display image: {e})")
        else:
            img_label.config(image="", text=f"(Image not found: {item.image_path})")

    def submit(label_choice: str):
        idx = state["index"]
        if idx >= len(pending):
            return
        item = pending[idx]
        rat = rationale_entry.get().strip()
        if not rat:
            rat = f"Observed {label_choice} failure."
        record = make_record(
            item=item,
            label=label_choice,
            rationale=rat,
            annotator=annotator,
        )
        append_annotation(record, output_path)
        state["annotated"] += 1
        state["index"] += 1
        display_current()

    # Buttons and bindings
    tk.Button(
        btn_frame, text="[S] Structural", bg="#e3f2fd", font=("Segoe UI", 10, "bold"),
        command=lambda: submit("structural"), width=15
    ).grid(row=0, column=0, padx=5)

    tk.Button(
        btn_frame, text="[F] Fabrication", bg="#fbe9e7", font=("Segoe UI", 10, "bold"),
        command=lambda: submit("fabrication"), width=15
    ).grid(row=0, column=1, padx=5)

    tk.Button(
        btn_frame, text="[A] Ambiguous", bg="#fff9c4", font=("Segoe UI", 10, "bold"),
        command=lambda: submit("ambiguous"), width=15
    ).grid(row=0, column=2, padx=5)

    root.bind("<Key-s>", lambda e: submit("structural"))
    root.bind("<Key-f>", lambda e: submit("fabrication"))
    root.bind("<Key-a>", lambda e: submit("ambiguous"))
    rationale_entry.bind("<Return>", lambda e: submit(state["selected_label"].get() or "structural"))

    display_current()
    root.mainloop()
    return state["annotated"]


def main():
    parser = argparse.ArgumentParser(description="Minimal blind annotation tool (P2.2).")
    parser.add_argument("input_file", help="Path to input items (.jsonl)")
    parser.add_argument(
        "--output", "-o", default="data/annotations/annotations.jsonl",
        help="Destination JSONL path for annotations (appended immediately per item)."
    )
    parser.add_argument("--annotator", "-a", default="Shrish", help="Annotator name or ID.")
    parser.add_argument("--image-dir", default=None, help="Base directory for relative image paths.")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Max items to annotate in this run.")
    parser.add_argument("--gui", action="store_true", help="Launch Tkinter GUI instead of CLI.")
    parser.add_argument("--no-open", action="store_true", help="In CLI mode, do not open image viewer.")

    args = parser.parse_args()

    items = load_items(args.input_file, image_base_dir=args.image_dir)
    out_path = pathlib.Path(args.output)

    if args.gui:
        run_gui(items, out_path, annotator=args.annotator, limit=args.limit)
    else:
        run_cli(items, out_path, annotator=args.annotator, limit=args.limit, open_image=not args.no_open)


if __name__ == "__main__":
    main()
