#!/usr/bin/env python3
"""
Generate all paper figures from reports/metrics.json + reports/benchmark.json.

This script is the single source of truth for the figures embedded in the paper.
Each figure is a pure function of the machine-readable evaluation artifacts
produced by scripts/evaluate.py and scripts/benchmark.py, so the figures are
always consistent with the numbers reported in the text/tables.

Figures produced (paper numbering):
  Fig 2 — token distribution by UPOS and error class
  Fig 3 — POS accuracy / Macro-F1 across configurations (ablation bar)
  Fig 4 — incremental ablation A→E (Macro-F1)
  Fig 5 — POS accuracy by category per config
  Fig 6 — confusion-matrix heatmap (best config)
  Fig 7 — throughput / RAM / accuracy 3-panel (Table 9 viz)
  Fig 8 — error-taxonomy F1 per class per config

Usage:
    .venv/bin/python scripts/make_figures.py [--reports reports] [--out reports/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# MDPI-friendly clean style.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Palette: one accent per pipeline configuration.
CONFIG_COLORS = {
    "[A] RMA-only": "#6b7280",
    "[B] RMA+CRF": "#3b82f6",
    "[C] RMA+Neural": "#10b981",
    "[D] Hybrid (cascade)": "#f59e0b",
    "[E] Hybrid+routing": "#ef4444",
    "[F] RMA+CRF+Neural-fallback": "#8b5cf6",
}
CLASS_COLORS = {
    "BOR": "#ef4444", "AFC": "#f59e0b", "SEG": "#10b981",
    "AGR": "#3b82f6", "NONE": "#9ca3af",
}
upos_order = ["NOUN", "VERB", "PUNCT", "ADJ", "PROPN", "PRON", "NUM",
              "ADV", "AUX", "CCONJ", "ADP", "DET", "PART", "SCONJ",
              "INTJ", "X", "SYM"]


def _load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    svg_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".svg")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")  # vector copy for the journal
    plt.close(fig)
    print(f"  wrote {path} + .svg")
    return path


# ---------------------------------------------------------------------------
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def fig_architecture(out_dir: str) -> None:
    """Fig 1: five-stage pipeline architecture with the three operating modes."""
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f8fafc", ec="#334155", fs=7.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, ls="-", color="#334155"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, color=color, lw=1.0,
                                     linestyle=ls, shrinkA=2, shrinkB=2))

    # ── Input / tokenizer / Stage 1 ──────────────────────────────────────
    box(0.15, 5.85, 1.5, 0.55, "Input text\n(Kazakh)", fc="#eef2ff")
    box(2.0, 5.85, 1.7, 0.55, "Tokenizer\n(script-aware)")
    box(4.05, 5.85, 3.1, 0.55, "Stage 1 · RMA\nreverse-Trie, 151 affixes, lexicons",
        fc="#eef2ff", bold=True)
    arrow(1.65, 6.12, 2.0, 6.12)
    arrow(3.7, 6.12, 4.05, 6.12)
    # Taxonomy side-signal from RMA
    box(7.5, 5.85, 2.35, 0.55, "Error taxonomy\nBOR/AFC/SEG/AGR/NONE",
        fc="#fef3c7", ec="#b45309")

    # ── Three operating modes (Stage 2/3 POS decision) ───────────────────
    modes = [
        ("Cascade (default)\nconfigs C–D", "#ecfdf5", "#059669",
         "Neural POS overrides;\nCRF bypassed when a\nneural model is loaded"),
        ("Routed (routing=True)\nconfig E", "#eff6ff", "#2563eb",
         "Confidence-weighted vote\nRMA+CRF+Neural; weights\nper error class (Table 4)"),
        ("Fallback (fallback=True)\nconfig F", "#f5f3ff", "#7c3aed",
         "CRF default POS; Neural\nonly if UNK/X or marginal\nconfidence < τ = 0.50"),
    ]
    xs = [0.4, 3.55, 6.7]
    for (title, fc, ec, body), x in zip(modes, xs):
        ax.add_patch(FancyBboxPatch((x, 3.9), 2.85, 1.35,
                                    boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.2))
        ax.text(x + 1.425, 5.02, title, ha="center", va="center",
                fontsize=7.4, fontweight="bold")
        ax.text(x + 1.425, 4.38, body, ha="center", va="center", fontsize=6.8)
        arrow(5.6, 5.85, x + 1.425, 5.25)

    # ── POS / lemma row ──────────────────────────────────────────────────
    box(0.4, 2.75, 4.35, 0.6, "POS decision  $p_i$  (per mode)", fc="#eef2ff", bold=True)
    box(5.3, 2.75, 4.35, 0.6, "Lemma  $l_i$:  neural lemmatizer (C–E)\n|  RMA lexicon (A, B, F)")
    for x in xs:
        arrow(x + 1.425, 3.9, 2.6 if x + 1.425 < 4.7 else 5.6, 3.35)
    arrow(4.75, 3.05, 5.3, 3.05)

    # ── Stage 4 / output ─────────────────────────────────────────────────
    box(0.4, 1.55, 9.25, 0.6, "Stage 4 · MFRN — 64 phonological constraints, 8 groups (diagnostic scoring)",
        fc="#eef2ff", bold=True)
    arrow(2.6, 2.75, 2.6, 2.15)
    arrow(7.5, 2.75, 7.5, 2.15)
    box(0.4, 0.35, 9.25, 0.7,
        "Output  $\\hat{y}_i = (p_i,\\ l_i,\\ e_i,\\ c_i)$   —   UPOS tag, lemma, error class, confidence",
        fc="#fef9c3", ec="#a16207", bold=True)
    arrow(5.0, 1.55, 5.0, 1.05)
    # Taxonomy feeds routing + output error class
    arrow(8.65, 5.85, 8.65, 5.25, ls="--", color="#b45309")
    arrow(8.65, 5.85, 9.55, 1.05, ls="--", color="#b45309")

    _save(fig, out_dir, "fig1_architecture.png")


def fig_class_distribution(metrics: dict, test_file: str, out_dir: str) -> None:
    """Fig 4: token distribution by predicted error class.

    Single panel (error classes only): the UPOS-distribution panel was
    dropped from the paper — it carries no argument that Table 2 and the
    per-tag F1 figure do not already make.
    """
    # Error-class distribution of the routed config. evaluate.py overwrites
    # every configuration's difficulty_class with the fixed standalone RMA
    # classification (θ=0.30), so this equals the model-independent
    # distribution of Table 2 / error_distribution.csv for any config.
    cfg = metrics["configurations"]
    name = next((k for k in cfg if "routing" in k), list(cfg)[-1])
    tax = cfg[name]["error_taxonomy_f1"]
    classes = ["BOR", "AFC", "SEG", "AGR", "NONE"]
    counts = [tax[c]["support"] for c in classes]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    total = sum(counts)
    bars = ax.bar(classes, counts,
                  color=[CLASS_COLORS[c] for c in classes], alpha=0.85)
    ax.set_title("Tokens by error class (predicted)")
    ax.set_ylabel("Token count")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{c/total*100:.1f}%", ha="center", va="bottom", fontsize=7)
    _save(fig, out_dir, "fig2_class_distribution.png")


def fig_ablation_macrof1(metrics: dict, out_dir: str) -> None:
    """Fig 3/4: Macro-F1 across ablation configurations A→E."""
    cfg = metrics["configurations"]
    names = list(cfg.keys())
    macro = [cfg[n]["macro_f1"] for n in names]
    acc = [cfg[n]["pos_accuracy"] / 100 for n in names]
    colors = [CONFIG_COLORS.get(n, "#999") for n in names]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, acc, w, label="POS accuracy", color=colors, alpha=0.55,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, macro, w, label="Macro-F1", color=colors, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("] ")[-1] for n in names], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("POS accuracy and Macro-F1 by pipeline configuration")
    ax.legend(loc="lower right")
    for i, (a, m) in enumerate(zip(acc, macro)):
        ax.text(i - w / 2, a + 0.01, f"{a*100:.1f}", ha="center", fontsize=7)
        ax.text(i + w / 2, m + 0.01, f"{m*100:.1f}", ha="center", fontsize=7)
    _save(fig, out_dir, "fig4_ablation_macrof1.png")


def fig_accuracy_by_category(metrics: dict, out_dir: str) -> None:
    """Fig 5: per-UPOS F1 across configurations."""
    cfg = metrics["configurations"]
    names = list(cfg.keys())
    upos = [t for t in upos_order
            if any(t in cfg[n]["per_tag_f1"] for n in names)]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(upos))
    n_cfg = len(names)
    w = 0.8 / max(n_cfg, 1)
    for i, name in enumerate(names):
        f1s = [cfg[name]["per_tag_f1"].get(t, {}).get("f1", 0.0) for t in upos]
        ax.bar(x + (i - n_cfg / 2) * w + w / 2, f1s, w,
               label=name.split("] ")[-1],
               color=CONFIG_COLORS.get(name, "#999"), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(upos, rotation=45, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-UPOS F1 across pipeline configurations")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    _save(fig, out_dir, "fig5_accuracy_by_category.png")


def fig_confusion_heatmap(cm_csv: str, out_dir: str) -> None:
    """Fig 6: confusion-matrix heatmap from confusion_matrix.csv."""
    if not os.path.exists(cm_csv):
        print(f"  (skip Fig 6: {cm_csv} not found)")
        return
    rows = []
    with open(cm_csv, encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        for r in rd:
            rows.append(r)
    tags = header[1:]
    mat = np.array([[int(v) for v in r[1:]] for r in rows], dtype=float)
    # Normalize by row (gold) so cell = recall into pred.
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sums, out=np.zeros_like(mat),
                     where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tags)))
    ax.set_yticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticklabels(tags)
    ax.set_xlabel("Predicted UPOS")
    ax.set_ylabel("Gold UPOS")
    ax.set_title("Confusion matrix (row-normalized recall)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir, "fig6_confusion_heatmap.png")


def fig_throughput_ram(bench: dict, out_dir: str) -> None:
    """Fig 7: throughput / RAM / accuracy 3-panel (Table 9 viz)."""
    if not bench:
        print("  (skip Fig 7: benchmark.json not found)")
        return
    cfgs = bench["configurations"]
    names = [c["configuration"] for c in cfgs]
    tps = [c["throughput_tokens_per_sec"] for c in cfgs]
    ram = [c["ram_mb"] for c in cfgs]
    acc = [c["accuracy_pct"] for c in cfgs]
    colors = [CONFIG_COLORS.get(n, "#999") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].bar(range(len(names)), acc, color=colors, alpha=0.85)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_title("(A) POS accuracy (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(acc):
        axes[0].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=7)

    axes[1].bar(range(len(names)), tps, color=colors, alpha=0.85)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_title("(B) Throughput (tok/s)")
    for i, v in enumerate(tps):
        axes[1].text(i, v + max(tps) * 0.02, f"{v:.0f}", ha="center", fontsize=7)

    axes[2].bar(range(len(names)), ram, color=colors, alpha=0.85)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=30, ha="right")
    axes[2].set_title("(C) RAM (MB, log scale)")
    axes[2].set_yscale("log")
    for i, v in enumerate(ram):
        axes[2].text(i, v * 1.1, f"{v:.0f}", ha="center", fontsize=7)
    _save(fig, out_dir, "fig7_throughput_ram.png")


def fig_taxonomy_f1(metrics: dict, out_dir: str) -> None:
    """Fig 8: per-error-class F1 across configurations."""
    cfg = metrics["configurations"]
    names = list(cfg.keys())
    classes = ["BOR", "AFC", "SEG", "AGR", "NONE"]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(classes))
    n_cfg = len(names)
    w = 0.8 / max(n_cfg, 1)
    for i, name in enumerate(names):
        f1s = [cfg[name]["error_taxonomy_f1"].get(c, {}).get("f1", 0.0)
               for c in classes]
        ax.bar(x + (i - n_cfg / 2) * w + w / 2, f1s, w,
               label=name.split("] ")[-1],
               color=CONFIG_COLORS.get(name, "#999"), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("POS accuracy within class")
    ax.set_ylim(0, 1.05)
    ax.set_title("POS accuracy by error-taxonomy class and configuration")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    _save(fig, out_dir, "fig8_taxonomy_f1.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test-file", default="data/test.conllu",
                    help="Used only for Fig 2 UPOS distribution if metrics.json "
                         "lacks it.")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.reports, "figures")
    os.makedirs(out_dir, exist_ok=True)

    metrics = _load(os.path.join(args.reports, "metrics.json"))
    bench = _load(os.path.join(args.reports, "benchmark.json"))
    cm_csv = os.path.join(args.reports, "confusion_matrix.csv")

    if not metrics:
        print(f"No metrics.json in {args.reports}. Run scripts/evaluate.py first.")
        return 1

    print(f"Generating figures into {out_dir} ...")
    fig_architecture(out_dir)
    fig_class_distribution(metrics, args.test_file, out_dir)
    fig_ablation_macrof1(metrics, out_dir)
    fig_accuracy_by_category(metrics, out_dir)
    fig_confusion_heatmap(cm_csv, out_dir)
    fig_throughput_ram(bench or {}, out_dir)
    fig_taxonomy_f1(metrics, out_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
