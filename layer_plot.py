import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from categories import CATEGORIES
from model_utils import get_num_blocks


DATA_DIR = "english/finegrained"
FNAME_TEMPLATE = "finegrained_blimp_{model_name}_1.0%.txt"
BLIMP_CATS = list(CATEGORIES["blimp"].keys())

MODELS = [
    ("GPT-2", "gpt2-xl"),
    ("Falcon", "Falcon3-3B-Base"),
    ("Llama", "Llama-3.2-3B"),
    ("Gemma", "gemma-3-4b-pt"),
    ("DeepSeek", "deepseek-llm-7b-base"),
    ("Phi", "Phi-4-mini-instruct"),
    ("Mistral", "Mistral-7B-v0.3"),
]
MODELS_SORTED = sorted(MODELS, key=lambda x: x[0].lower())


suite2cat = {}
for cat, suites in CATEGORIES["blimp"].items():
    for s in suites:
        suite2cat[s] = cat


def parse_model_file(path):
    cat_units = defaultdict(set)
    current_suite = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts[0].isdigit():
                current_suite = parts[0]
                continue

            _, layer_str, block_type, pos_str = parts
            layer_idx = int(layer_str)
            pos_idx = int(pos_str)

            if current_suite in suite2cat:
                cat = suite2cat[current_suite]
                cat_units[cat].add((layer_idx, block_type, pos_idx))

    return cat_units


n_models = len(MODELS_SORTED)

category_counts = {cat: [None] * n_models for cat in BLIMP_CATS}
model_num_layers = []

for m_idx, (pretty_name, model_name) in enumerate(MODELS_SORTED):
    fname = FNAME_TEMPLATE.format(model_name=model_name)
    path = os.path.join(DATA_DIR, fname)
    print(f"Reading {path}")

    cat_units = parse_model_file(path)
    n_layers = get_num_blocks(model_name)
    model_num_layers.append(n_layers)

    for cat in BLIMP_CATS:
        units = cat_units.get(cat, set())
        if not units:
            counts = np.zeros(n_layers, dtype=int)
        else:
            counts = np.zeros(n_layers, dtype=int)
            for layer_idx, block_type, pos_idx in units:
                if 0 <= layer_idx < n_layers:
                    counts[layer_idx] += 1
        category_counts[cat][m_idx] = counts


n_rows = 6
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 20), sharex=True, sharey=True)

model_labels = [pretty for pretty, _ in MODELS_SORTED]

x_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

last_im = None

for i, cat in enumerate(BLIMP_CATS):
    ax = axes[i // n_cols, i % n_cols]
    norm = LogNorm(vmin=1e-2, vmax=100.0)
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad(color="white")

    for m_idx in range(n_models):
        counts = category_counts[cat][m_idx]
        if counts is None:
            continue
        total = float(np.nansum(counts))
        if total <= 0:
            arr = np.zeros((1, counts.shape[0]), dtype=float)
        else:
            arr = (counts / total * 100.0)[np.newaxis, :]
        arr[arr == 0] = np.nan
        extent = [0.0, 1.0, m_idx - 0.5, m_idx + 0.5]
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            norm=norm,
            cmap=cmap,
            extent=extent,
        )
        last_im = im

    ax.set_yticks(np.arange(n_models))
    ax.set_yticklabels(model_labels)

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x:.2g}" for x in x_ticks])

    for xt in x_ticks:
        ax.axvline(xt, linewidth=0.3, color="k", alpha=0.2)

    pretty_cat = cat.replace("_", " ")
    if pretty_cat.lower() == "npi licensing":
        pretty_cat = "NPI Licensing"
    elif pretty_cat.lower() == "subject verb agreement":
        pretty_cat = "S-V Agreement"
    elif pretty_cat.lower() == "determiner noun agreement":
        pretty_cat = "DET-N Agreement"
    elif pretty_cat.lower() == "control raising":
        pretty_cat = "Control/Raising"
    elif pretty_cat.lower() == "filler gap":
        pretty_cat = "Filler-Gap"
    else:
        pretty_cat = pretty_cat.title()
    ax.set_title(pretty_cat, fontsize=10)

    if i // n_cols == n_rows - 1:
        ax.set_xlabel("Relative layer depth")
    if i % n_cols == 0:
        ax.set_ylabel("Model")

cax = fig.add_axes([0.88, 0.4, 0.02, 0.15])
cbar = fig.colorbar(last_im, cax=cax)
cbar.set_label("Percentage of units")
cbar.set_ticks([0.1, 1, 10, 100])

fig.suptitle(
    "Relative Depth Breakdown of Responsive Units", fontsize=14, y=0.96, x=0.49
)
fig.tight_layout(rect=[0, 0, 0.85, 0.96])
plt.savefig("english/finegrained/layers.pdf")
plt.savefig("english/finegrained/layers.png", dpi=300)
