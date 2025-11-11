import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr
import numpy as np


p = argparse.ArgumentParser()
p.add_argument("--ablation-dir", type=str, default="english/ablation")
p.add_argument("--cv-dir", type=str, default="english/cross-validation")
p.add_argument("--dataset", type=str, default="BLiMP")
p.add_argument("--title", type=str, default="Ablation effect vs. 2-fold overlap")
args = p.parse_args()

ablation_dir = args.ablation_dir
cv_dir = args.cv_dir
dataset = args.dataset
title = args.title


ablation_files = [
    f
    for f in os.listdir(ablation_dir)
    if f.startswith(f"ablation_{dataset.lower()}_") and f.endswith(".txt")
]
cv_files = [
    f
    for f in os.listdir(cv_dir)
    if f.startswith(f"cross-validation_{dataset.lower()}_") and f.endswith(".txt")
]

percentage = None

ablation_effects_all = {}
cv_scores_all = {}


for ablation_file in ablation_files:
    ablation_filepath = os.path.join(ablation_dir, ablation_file)
    ablation_parts = ablation_file.split(
        "_"
    )  # ["ablation", dataset, model_name, percentage]
    assert len(ablation_parts) == 4
    percentage = ablation_parts[3][:-4]

    print(ablation_file)

    model_name = ablation_parts[2]
    cv_file = None
    for f in cv_files:
        if f"_{model_name}_" in f:
            cv_file = f
            break
    cv_filepath = os.path.join(cv_dir, cv_file)

    print(cv_file)

    ablation_effects = {}
    with open(ablation_filepath, "r") as f:
        for line in f:
            parts = line.strip().split()  # [f"{suite}:", f"{original}/{top}/{random}"]
            if len(parts) == 2:
                suite, nums = parts
                original, top, random = nums.split("/")
                ablation_effects[suite[:-1]] = float(top) - float(original)

    cv_scores = {}
    with open(cv_filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # [num_units, f"({percentage}%)", suite]
                cv_scores[parts[2]] = float(parts[1][1:-2])

    for suite, effect in ablation_effects.items():
        if suite in cv_scores:
            if suite not in ablation_effects_all:
                ablation_effects_all[suite] = []
                cv_scores_all[suite] = []
            ablation_effects_all[suite].append(effect)
            cv_scores_all[suite].append(cv_scores[suite])


x = []
y = []
for suite in ablation_effects_all:
    if suite in cv_scores_all:
        avg_cv = np.mean(cv_scores_all[suite])
        avg_ablation = np.mean(ablation_effects_all[suite])
        x.append(avg_cv)
        y.append(avg_ablation)

r_value, p_value = pearsonr(x, y)


fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x, y, color="blue", alpha=0.8, edgecolor="black", linewidth=0.5, marker="o")

ax.tick_params(axis="x", labelsize=18)
ax.set_xlabel("Cross-Validation Consistency", fontsize=18)

ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0, symbol=None))
ax.set_yticks(np.arange(-0.15, 0.1, 0.05))
ax.tick_params(axis="y", labelsize=18)
ax.set_ylabel(f"Top-{percentage} Ablation Effect", fontsize=18)

ax.set_title(f"{title} (R = {r_value:.3f}, p = {p_value:.3f})", fontsize=20)

ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.commonprefix([ablation_dir, cv_dir]) + "scatterplot.pdf")
plt.savefig(os.path.commonprefix([ablation_dir, cv_dir]) + "scatterplot.png")
