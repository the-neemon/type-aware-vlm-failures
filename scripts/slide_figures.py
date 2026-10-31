"""Figures for the final slides, generated from results/ only.

    python scripts/slide_figures.py

Writes paper/final/figures/slide_*.pdf. Colours: Qwen blue #2a78d6, LLaVA orange
#eb6834 (the validated pair the report figures use), baselines grey; every bar
carries its value, so identity never rests on colour alone.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                            "font.family": "sans-serif", "axes.spines.top": False,
                            "axes.spines.right": False})
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/final/figures"
P = ROOT / "results/probes"
MODELS = {"qwen2_5_vl_7b": ("Qwen2.5-VL-7B", "#2a78d6"),
          "llava_next_mistral_7b": ("LLaVA-NeXT-7B", "#eb6834")}
GREY = "#9a9a9a"


def load(name):
    return json.loads((P / name).read_text(encoding="utf-8"))


def bars(ax, groups, series, colors, ylim=(0.4, 1.0), chance=True):
    """Grouped bars with value labels. series: {label: [value per group or None]}."""
    n = len(series)
    width = 0.8 / n
    for i, (label, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        for x, v in zip(xs, vals):
            if v is None:
                continue
            ax.bar(x, v - ylim[0], width * 0.92, bottom=ylim[0], color=colors[i],
                   label=label if x == xs[0] or vals[0] is None else None)
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(*ylim)
    ax.set_ylabel("AUROC")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    if chance and ylim[0] <= 0.5:
        ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":")


def e1_controls():
    """E1 binary probe: reported split, five-split mean, residualised, surface only."""
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    groups, series = [], {"Probe (reported split)": [], "Mean of 5 splits": [],
                          "Surface features removed": [], "Surface features only": []}
    for m, (name, _) in MODELS.items():
        r = load(f"binary_followups_{m}.json")
        s42 = r["seeds"]["42"]
        groups.append(name)
        series["Probe (reported split)"].append(s42["plain"]["test_auroc"])
        series["Mean of 5 splits"].append(r["over_seeds"]["plain"]["mean"])
        series["Surface features removed"].append(s42["residualised"]["test_auroc"])
        series["Surface features only"].append(
            load(f"binary_{m}_test.json")["surface_baseline"]["auroc"])
    bars(ax, groups, series, ["#2a78d6", "#1baf7a", "#eda100", GREY])
    fig.legend(*ax.get_legend_handles_labels(), fontsize=7, frameon=False, ncol=2,
               loc="lower center")
    ax.set_title("Predicting a wrong answer (ChartQA test, all errors)", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "slide_e1_controls.pdf")
    plt.close(fig)


def chartqa_types():
    """E3 on ChartQA: each probe, its question-type and residualised variants, surface."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True)
    contrasts = [("structural", "S_vs_C", "retrieval", "Misread\nvs correct"),
                 ("computation", "K_vs_C", "arithmetic", "Arithmetic error\nvs correct"),
                 ("struct_vs_comp", "S_vs_K", None, "Misread vs\narithmetic error")]
    for ax, (m, (name, color)) in zip(axes, MODELS.items()):
        full = load(f"chartqa_types_{m}.json")["probes"]
        resid = load(f"chartqa_types_residualised_{m}.json")["probes"]
        series = {"Probe": [], "One question type only": [],
                  "Surface features removed": [], "Surface features only": []}
        for probe, target, qtype, _ in contrasts:
            series["Probe"].append(full[probe]["matrix"][target]["auroc"])
            series["One question type only"].append(
                load(f"chartqa_types_{qtype}_{m}.json")["probes"][probe]["matrix"][target]["auroc"]
                if qtype else None)
            series["Surface features removed"].append(
                resid[probe]["residualised"]["matrix"][target]["auroc"])
            series["Surface features only"].append(
                full[probe]["surface_baseline"]["matrix"][target]["auroc"])
        bars(ax, [c[3] for c in contrasts], series, [color, "#1baf7a", "#eda100", GREY])
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "slide_chartqa_types.pdf")
    plt.close(fig)


def pilot3_layers():
    """Missing-vs-shown probe by depth: on pilots 1-2 (training, out of fold) and pilot 3."""
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for m, (name, color) in MODELS.items():
        e = load(f"pilot3_{m}.json")["probes"]["missing_vs_shown"]
        train = {int(k): v for k, v in e["train_oof_auroc_by_layer"].items()}
        p3 = {int(k): v for k, v in e["pilot3_absent_vs_shown_by_layer_descriptive_only"].items()}
        n = max(train) + 1
        ax.plot([k / (n - 1) for k in sorted(train)], [train[k] for k in sorted(train)],
                color=color, linestyle="--", linewidth=1.5,
                label=f"{name}: missing names (pilots 1-2)")
        ax.plot([k / (n - 1) for k in sorted(p3)], [p3[k] for k in sorted(p3)],
                color=color, linewidth=2, label=f"{name}: out-of-range years (pilot 3)")
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Relative depth (layer / last layer)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.grid(color="#e6e6e6", linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "slide_pilot3_layers.pdf")
    plt.close(fig)


def gridline():
    """Synthetic misread vs correct: probe vs chart-metadata baselines."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    groups, series = [], {"Activation probe": [], "Chart metadata": [],
                          "+ gridline distance (declared)": [],
                          "+ relative distance (post hoc)": []}
    for m, (name, _) in MODELS.items():
        g = load(f"synth_grid_baseline_{m}.json")
        cv = g["cross_validated"]
        groups.append(name)
        series["Activation probe"].append(g["activation_probe_S_vs_C"]["auroc"])
        series["Chart metadata"].append(cv["metadata"]["S_vs_C"])
        series["+ gridline distance (declared)"].append(cv["metadata_plus_grid"]["S_vs_C"])
        series["+ relative distance (post hoc)"].append(
            cv["post_hoc_metadata_plus_all_grid"]["S_vs_C"])
    bars(ax, groups, series, ["#2a78d6", GREY, "#1baf7a", "#eda100"])
    fig.legend(*ax.get_legend_handles_labels(), fontsize=7, frameon=False, ncol=2,
               loc="lower center")
    ax.set_title("Synthetic misreads vs correct reads", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "slide_gridline.pdf")
    plt.close(fig)


def risk_coverage():
    """E5 risk-coverage at slide size, five policies (results/e5/*_e5.json)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from final_assets import e5_results   # sets serif fonts on import; restore ours
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    res = e5_results()
    styles = [("plain", "Plain model (answer everything)", "#9a9a9a", ":"),
              ("binary", "Binary abstention", "#2a78d6", "-"),
              ("controller", "Controller", "#eb6834", "-"),
              ("relook_binary", "Re-look + binary", "#1baf7a", "-"),
              ("oracle", "Oracle (knows the true type)", "black", "--")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (m, (name, _)) in zip(axes, MODELS.items()):
        for p, label, color, ls in styles:
            risk = res[m]["policies"][p]["curve"]
            n = len(risk)
            ax.plot([(i + 1) / n for i in range(n)], risk, color=color, ls=ls, lw=1.8,
                    label=label)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Coverage (fraction of questions answered)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="#e6e6e6", linewidth=0.6)
    axes[0].set_ylabel("Risk (error rate of answered)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "slide_risk_coverage.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (e1_controls, chartqa_types, pilot3_layers, gridline, risk_coverage):
        f()
    print("wrote", *sorted(p.name for p in OUT.glob("slide_*.pdf")))


if __name__ == "__main__":
    main()
