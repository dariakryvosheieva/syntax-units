import glob
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from categories import CATEGORIES
from plot_utils import pretty_category_name


Unit = Tuple[int, int, str, int]


MODEL_COLORS = [
    "#e41a1c",
    "#3faa3b",
    "#923f9e",
    "#fc7cbf",
    "#e4d100",
    "#9d4c1d",
    "#999999",
]


def parse_units_file(path: str) -> Dict[str, List[Unit]]:
    phenomena: Dict[str, List[Unit]] = {}
    current: Optional[str] = None
    unit_line_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]+)\s+(\d+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = unit_line_re.match(line)
            if m is None:
                current = line
                if current not in phenomena:
                    phenomena[current] = []
                continue
            if current is None:
                current = "UNKNOWN"
                if current not in phenomena:
                    phenomena[current] = []
            idx = int(m.group(1))
            layer_idx = int(m.group(2))
            kind = m.group(3).lower()
            idx_within_layer = int(m.group(4))
            phenomena[current].append((idx, layer_idx, kind, idx_within_layer))
    return phenomena


def compute_layer_shares_for_file(path: str):
    ph = parse_units_file(path)

    max_layer = -1
    for items in ph.values():
        for _, layer_idx, kind, _ in items:
            if kind in ("mlp", "attn"):
                if layer_idx > max_layer:
                    max_layer = layer_idx
    L = max_layer + 1 if max_layer >= 0 else 0
    _, dataset, model_name, perc_str = path.split("_")

    if L <= 0:
        return {"model": model_name, "x": np.array([]), "L": 0, "per_ph": {}}

    per_ph: Dict[str, np.ndarray] = {}
    for name, items in ph.items():
        counts = np.zeros(L, dtype=int)
        total = 0
        for _, layer_idx, kind, _ in items:
            if kind in ("mlp", "attn"):
                if 0 <= layer_idx < L:
                    counts[layer_idx] += 1
                    total += 1
        if total > 0:
            per_ph[name] = counts.astype(float) / float(total)

    x = (np.arange(L, dtype=float) + 0.5) / float(L)
    return {"model": model_name, "x": x, "L": L, "per_ph": per_ph}


def plot_blimp_categories_layer_shares(
    models_data: List[dict],
    categories: Dict[str, List[str]],
):
    colors = {}
    for i, md in enumerate(models_data):
        colors[md["model"]] = MODEL_COLORS[i % len(MODEL_COLORS)]

    cat_names = list(categories.keys())
    fig, axes = plt.subplots(6, 2, figsize=(10, 25), sharey=True)
    axes = axes.ravel()

    model_legend_handles = {}

    all_phens = [p for plist in categories.values() for p in plist]

    for ai, cat in enumerate(cat_names):
        ax = axes[ai]
        phens = categories[cat]

        for md in models_data:
            model = md["model"]
            x = md["x"]
            if x.size == 0:
                continue
            for ph in phens:
                y = md["per_ph"].get(ph)
                if y is None:
                    continue
                (line,) = ax.plot(
                    x,
                    y,
                    linewidth=1.2,
                    alpha=0.9,
                    color=colors[model],
                )
                if model not in model_legend_handles:
                    model_legend_handles[model] = line

        ax.set_title(pretty_category_name(cat), fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 0.85)
        ax.set_xlabel("Relative Depth")
        if ai % 4 == 0:
            ax.set_ylabel("Share of selective units")
        ax.grid(True, axis="both", linestyle=":", alpha=0.35)

    handles = [model_legend_handles[m] for m in sorted(model_legend_handles)]
    labels = sorted(model_legend_handles)
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.08, 0.91),
        frameon=True,
        title="Models",
    )

    fig.suptitle("Relative Depth Breakdown of Responsive Units", fontsize=14, x=0.45)
    fig.tight_layout(rect=[0, 0, 0.88, 0.98])

    fig.savefig("english/finegrained/layers.png", bbox_inches="tight")
    fig.savefig("english/finegrained/layers.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    file_paths = sorted(glob.glob("english/finegrained/*.txt"))

    models_data = [compute_layer_shares_for_file(p) for p in file_paths]

    plot_blimp_categories_layer_shares(
        models_data=models_data,
        categories=CATEGORIES["blimp"],
    )


if __name__ == "__main__":
    main()
