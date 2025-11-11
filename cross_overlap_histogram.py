import argparse
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from categories import CATEGORIES
from plot_utils import get_canonical_order, load_model_matrices


def build_suite_to_category(category_map: Dict[str, List[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for cat, suites in category_map.items():
        for s in suites:
            mapping[s] = cat
    return mapping


def compute_pairwise_overlaps(
    matrices: Dict[str, pd.DataFrame],
    category_map: Dict[str, List[str]],
    canonical_order: List[str],
) -> pd.DataFrame:
    suite_to_cat = build_suite_to_category(category_map)
    acc: Dict[Tuple[str, str], List[float]] = {}

    for df in matrices.values():
        for i_idx, i in enumerate(canonical_order):
            for j in canonical_order[i_idx + 1 :]:
                val = float(df.at[i, j])  # A-B == B-A
                key = (i, j) if i < j else (j, i)
                acc.setdefault(key, []).append(val)

    rows = []
    for (i, j), vals in acc.items():
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "suite1": i,
                "suite2": j,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "same_category": suite_to_cat.get(i) == suite_to_cat.get(j),
                "category1": suite_to_cat.get(i, ""),
                "category2": suite_to_cat.get(j, ""),
                "n_models": int(arr.size),
            }
        )

    df_pairs = (
        pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    )
    print(df_pairs.head()[["suite2", "mean"]])
    print(df_pairs.tail()[["suite2", "mean"]])
    return df_pairs


def plot_pairwise_histogram(
    data: pd.DataFrame,
    dataset: str,
    directory: str,
):
    bar_h = 0.005
    fig_w = 10
    fig_h = bar_h * len(data)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = np.where(data.same_category.values, "#89CCF1", "#FFB668")
    y = np.arange(len(data)) * bar_h
    ax.barh(y, data["mean"].values, color=colors, height=bar_h, zorder=2)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_ylim(-bar_h / 2.0, len(data) * bar_h - bar_h / 2.0)
    ax.margins(y=0)
    ax.invert_yaxis()

    ax.set_xlabel("Percentage of units", fontsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlim(0, 90)

    ax.set_title(
        f"Pairwise unit overlaps for all pairs of {dataset} phenomena", fontsize=18
    )

    h_same = Patch(facecolor="#89CCF1", edgecolor="black")
    h_diff = Patch(facecolor="#FFB668", edgecolor="black")
    
    ax.legend(
        handles=[h_same, h_diff],
        labels=["Same category", "Different categories"],
        loc="lower right",
        frameon=True,
        fontsize=15,
    )

    topk = data.nlargest(10, "mean").reset_index(drop=True)

    y0 = -bar_h / 2.0
    h_rect = len(topk) * bar_h
    x0 = 70
    w_rect = 18
    
    topbars_rect = plt.Rectangle(
        (x0, y0),
        w_rect,
        h_rect,
        fill=False,
        linewidth=1.5,
        edgecolor="black",
        zorder=5,
        transform=ax.transData,
    )
    ax.add_patch(topbars_rect)

    callout_ax = inset_axes(
        ax,
        width="54%",
        height="42%",
        loc="upper center",
        bbox_to_anchor=(0.17, -0.07, 0.95, 0.99),
        bbox_transform=fig.transFigure,
        borderpad=0.0,
    )
    callout_ax.set_facecolor("white")

    for spine in callout_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")

    callout_ax.set_xticks([])
    callout_ax.set_yticks([])
    callout_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    n = len(topk)
    top_margin = 0.03
    bottom_margin = 0.03
    usable_h = 1.0 - top_margin - bottom_margin
    row_h = usable_h / n
    dy = row_h * 0.2
    x_pct = 0.14
    x_names = 0.17

    for i, r in topk.iterrows():
        y_center = 1.0 - top_margin - (i + 0.5) * row_h

        pct = f"{r['mean']:.1f}%"
        callout_ax.text(
            x_pct,
            y_center,
            pct,
            ha="right",
            va="center",
            fontsize=11,
            transform=callout_ax.transAxes,
        )

        s1 = f"{r['suite1']} /"
        s2 = f"{r['suite2']}"
        callout_ax.text(
            x_names,
            y_center + dy,
            s1,
            ha="left",
            va="center",
            fontsize=11,
            transform=callout_ax.transAxes,
        )
        callout_ax.text(
            x_names,
            y_center - dy,
            s2,
            ha="left",
            va="center",
            fontsize=11,
            transform=callout_ax.transAxes,
        )

    x_bar_mid = x0 + w_rect / 2.0
    y_bar_bottom = y0 + h_rect

    x_callout_top = 0.5
    y_callout_top = 1.0

    connector = ConnectionPatch(
        xyA=(x_bar_mid, y_bar_bottom),
        coordsA=ax.transData,
        xyB=(x_callout_top, y_callout_top),
        coordsB=callout_ax.transAxes,
        linewidth=1,
        color="black",
        zorder=7,
    )
    fig.add_artist(connector)

    fig.tight_layout()
    fig.savefig(f"{directory}/cross_overlap_histogram_{dataset.lower()}.png", dpi=200)
    fig.savefig(f"{directory}/cross_overlap_histogram_{dataset.lower()}.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP")
    p.add_argument("--directory", default="english/cross-overlap")
    args = p.parse_args()

    category_map = CATEGORIES[args.dataset.lower()]
    canonical_order = get_canonical_order(category_map)

    matrices = load_model_matrices(
        args.directory,
        args.dataset.lower(),
        args.dataset.lower(),
        canonical_order,
        canonical_order,
    )
    df_pairs = compute_pairwise_overlaps(matrices, category_map, canonical_order)

    plot_pairwise_histogram(
        df_pairs,
        dataset=args.dataset,
        directory=args.directory,
    )


if __name__ == "__main__":
    main()
