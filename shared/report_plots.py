"""Figure templates for the report. Owner: Shi Ting.

Each function takes a tidy table produced by step 6 and returns a matplotlib Figure.
save() writes PNG (for the report) and PDF (vector, for print).
Palette: fixed categorical order, never cycled; diverging red <-> grey <-> blue for deltas.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("g11_div", ["#e34948", "#f0efec", "#2a78d6"])
KIND_COLOR = {"baseline": SERIES[1], "main": SERIES[0], "reference": SERIES[2], "ablation": MUTED}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
        "font.family": "sans-serif",
    })


def save(fig, path_stem: str) -> list[str]:
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    paths = [f"{path_stem}.png", f"{path_stem}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def forest_plot(deltas: pd.DataFrame, label: str = "model", metric: str = "metric",
                title: str = "Change against reference, 95% user-clustered interval"):
    """One panel per metric; one row per model/variant: point = delta, bar = interval.

    deltas: long table with columns [label, metric, delta, ci_low, ci_high].
    """
    metrics = list(dict.fromkeys(deltas[metric]))
    labels = list(dict.fromkeys(deltas[label]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 0.38 * len(labels) + 1.2),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], metrics):
        d = deltas[deltas[metric] == m].set_index(label).reindex(labels)
        yy = np.arange(len(labels))[::-1]
        sig = (d["ci_low"] > 0) | (d["ci_high"] < 0)
        ax.hlines(yy, d["ci_low"], d["ci_high"], color=np.where(sig, SERIES[0], MUTED), lw=2)
        ax.scatter(d["delta"], yy, s=36, color=np.where(sig, SERIES[0], MUTED), zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(yy, labels)
        ax.set_title(m, fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, -0.02, "Blue: interval excludes zero. Grey: no detectable difference.", fontsize=8, color=MUTED)
    fig.tight_layout()
    return fig


def tradeoff_plot(points: pd.DataFrame, curves: dict | None = None, x: str = "warm_recall@10",
                  y: str = "cold_recall@10", label: str = "model", kind: str = "kind",
                  title: str = "Cold vs warm Recall@10"):
    """Scatter of models (coloured by kind) plus optional slot-quota curves.

    points: one row per model with x, y, label and kind (baseline/main/reference/ablation).
    curves: {name: table indexed by quota with x and y columns}.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i, (name, c) in enumerate((curves or {}).items()):
        col = SERIES[i % 3]
        ax.plot(c[x], c[y], "-", color=col, lw=1.5, alpha=0.8, zorder=1)
        if i == 0:   # quota labels on the first (main) curve only, to avoid clutter
            for q, row in c.iterrows():
                if q > 0:
                    ax.annotate(f"q={q:.0%}", (row[x], row[y]), xytext=(5, -3), textcoords="offset points",
                                fontsize=7, color=INK2)
        ax.plot([], [], "-", color=col, label=f"slot quota: {name}")
    order = ["main", "reference", "baseline", "ablation"]
    for k in sorted(points[kind].unique(), key=lambda v: order.index(v) if v in order else 9):
        d = points[points[kind] == k].sort_values(y)
        ax.scatter(d[x], d[y], s=48 if k != "ablation" else 26, color=KIND_COLOR.get(k, MUTED),
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=k)
        for j, (_, row) in enumerate(d.iterrows()):
            name = str(row[label]).replace("deep_", "") if k == "ablation" else str(row[label])
            dy = (4, -10)[j % 2] if k == "ablation" else 5
            ax.annotate(name, (row[x], row[y]), xytext=(5, dy), textcoords="offset points",
                        fontsize=7, color=INK2 if k != "ablation" else MUTED)
    ax.set_xlabel(f"{x} (warm articles)")
    ax.set_ylabel(f"{y} (cold articles)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def metric_by_bucket(table: pd.DataFrame, ylabel: str = "Recall@10", title: str = "Recall@10 by prior-exposure bucket",
                     support: pd.Series | None = None):
    """Lines across ordered buckets; columns are models (<= 3 for colour; more go grey)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.arange(len(table.index))
    for i, col in enumerate(table.columns):
        c = SERIES[i] if i < 3 else MUTED
        ax.plot(xs, table[col], "-o", color=c, ms=5, label=col, mec=SURFACE, mew=1.5)
    labels = [str(b) for b in table.index]
    if support is not None:
        labels = [f"{b}\n(n={int(support.get(b, 0)):,})" for b in table.index]
    ax.set_xticks(xs, labels)
    ax.set_xlabel("prior exposures at impression time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def ablation_heatmap(matrix: pd.DataFrame, lo: pd.DataFrame | None = None, hi: pd.DataFrame | None = None,
                     title: str = "Change in Recall@10 when a tier is removed, by exposure bucket"):
    """Rows = variants (tier removed), columns = exposure buckets, cells = delta vs full.

    Red = removing the tier lowers recall (the tier helps). Cells whose interval
    excludes zero are marked with a dot when lo/hi are given.
    """
    v = matrix.to_numpy(np.float64)
    lim = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.1 * matrix.shape[1] + 2.2, 0.55 * matrix.shape[0] + 1.4))
    im = ax.imshow(v, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim or -1, lim or 1), aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index])
    ax.grid(False)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                sig = lo is not None and (lo.iloc[i, j] > 0 or hi.iloc[i, j] < 0)
                ax.text(j, i, f"{v[i, j]:+.3f}{' •' if sig else ''}", ha="center", va="center", fontsize=7,
                        color=INK)
    ax.set_xlabel("prior exposures at impression time")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    fig.tight_layout()
    return fig


def threshold_figure(noise: pd.DataFrame, volume: pd.DataFrame, thresholds=(5, 10, 20)):
    """Left: early-CTR error vs prior exposures. Right: hours to reach each threshold."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    a1.plot(noise.index, noise["mae"], "-o", color=SERIES[0], ms=4, mec=SURFACE, label="observed error")
    a1.plot(noise.index, noise["binomial_sd"], "--", color=MUTED, lw=1.5, label="binomial SD at global CTR")
    for t in thresholds:
        a1.axvline(t, color=AXIS, lw=1, ls=":")
        a1.text(t, a1.get_ylim()[1], f" {t}", color=INK2, fontsize=8, va="top")
    a1.set_xscale("log")
    a1.set_xlabel("prior exposures n")
    a1.set_ylabel("|CTR after n − later CTR|")
    a1.set_title("Noise in early CTR")
    a1.legend()
    xs = np.arange(len(volume.index))
    a2.bar(xs, volume["median_hours_to_t"], width=0.5, color=SERIES[0])
    for i, (t, row) in enumerate(volume.iterrows()):
        a2.text(i, row["median_hours_to_t"], f"{row['median_hours_to_t']:.1f} h\n{row['share_reaching_t']:.0%} reach",
                ha="center", va="bottom", fontsize=8, color=INK2)
    a2.set_xticks(xs, [f"t = {t}" for t in volume.index])
    a2.set_ylabel("median hours from first exposure")
    a2.set_title("Time to leave the cold state")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
