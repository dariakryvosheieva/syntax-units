import os, argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from categories import CATEGORIES
from model_utils import get_num_blocks, get_hidden_dim
from plot_utils import pretty_category_name, CATEGORY_COLORS
from utils import random_overlap_expected


def read_control_file(path: str, expect_many: bool):
    records = []
    with open(path) as f:
        for line in f:
            spl = line.strip().split()
            if len(spl) < 1:
                continue
            records.append(int(spl[0]))
    if not records:
        raise RuntimeError(f"{path} is empty?")
    avg = float(np.mean(records))
    df = pd.DataFrame({"LangOverlap": records}) if expect_many else None
    return avg, df


def add_score(table, model_name, suite, score):
    if suite not in table:
        table[suite] = {}
    table[suite][model_name] = score


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP")
    p.add_argument("--directory", default=f"english/cross-validation")
    p.add_argument("--title", default=f"Average 2-fold overlap")
    args = p.parse_args()

    model_names = set()
    suite_rows = {}
    percentage_units = {}

    num = 0
    sum = 0

    for fname in os.listdir(args.directory):
        if fname.startswith(
            f"cross-validation_{args.dataset.lower()}_"
        ) and fname.endswith("txt"):
            _, dataset, model_name, perc_str, num_folds = fname.split("_")
            percent = float(perc_str[:-1]) / 100
            model_names.add(model_name)

            print(fname)

            n_units_total = get_num_blocks(model_name) * get_hidden_dim(model_name)
            k_units = int(n_units_total * percent)

            percentage_units[model_name] = k_units

            with open(os.path.join(args.directory, fname)) as fh:
                for ln in fh:
                    p = ln.strip().split()
                    if len(p) != 3:
                        continue
                    sum += float(p[0]) / k_units
                    num += 1
                    add_score(suite_rows, model_name, p[2], float(p[0]))

            for ctrl_file in os.listdir(args.directory):
                if f"cross-validation_blimp-control_{model_name}" in ctrl_file:
                    ctrl_path = os.path.join(args.directory, ctrl_file)
                    score, _ = read_control_file(ctrl_path, expect_many=False)
                    add_score(
                        suite_rows, model_name, "BLiMP-Control (Avg.)", float(score)
                    )

            rand_score = random_overlap_expected(
                n_units_total, k_units, n_folds=int(num_folds[:-9])
            )
            add_score(suite_rows, model_name, "Random", float(rand_score))

    model_list = sorted(model_names)

    avg_entries = {
        suite: np.mean(
            [100 * individual_scores[m] / percentage_units[m] for m in model_list]
        )
        for suite, individual_scores in suite_rows.items()
    }

    print(
        "Average overlap: ",
        np.mean(
            [
                avg_entries[suite]
                for suite in avg_entries
                if "Control" not in suite and "Random" not in suite
            ]
        ),
    )

    category_map = {}
    for cat, subdict in CATEGORIES[args.dataset.lower()].items():
        for suite_name in subdict:
            category_map[suite_name] = cat

    unique_cats = sorted(set(category_map[s] for s in avg_entries if s in category_map))

    cat2color = {
        cat: CATEGORY_COLORS[args.dataset.lower()][i]
        for i, cat in enumerate(unique_cats)
    }

    GREY_CTRL = "#888888"
    GREY_RAND = "#bbbbbb"

    avg_entries = dict(sorted(avg_entries.items(), key=lambda kv: kv[1], reverse=True))

    n_bars = len(avg_entries)
    bar_h = 1.0
    fig, ax = plt.subplots(figsize=(8, n_bars * 0.18))

    y_pos = np.arange(n_bars)

    for i, (suite, avg_score) in enumerate(avg_entries.items()):
        if suite.startswith("BLiMP-Control"):
            color = GREY_CTRL
        elif suite == "Random":
            color = GREY_RAND
        else:
            color = cat2color.get(category_map[suite], "#1f77b4")

        ax.barh(
            y_pos[i],
            avg_score,
            height=bar_h,
            color=color,
            edgecolor="black",
        )

        ax.text(103, y_pos[i], f"{avg_score:.2f}%", va="center", fontsize=10)

        if len(model_list) > 1:
            for j, model in enumerate(model_list):
                score = suite_rows[suite][model] / percentage_units[model] * 100
                offset = (j - 3.5) * 0.04
                ax.scatter(score, y_pos[i] + offset, color="black", marker="o", s=10)

    ax.set_xlabel("Percentage of Units", fontsize=12)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=10)

    ax.set_yticks([])
    ax.invert_yaxis()

    ax.set_title(args.title, fontsize=14)

    handles = (
        [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=cat2color[cat],
                markeredgecolor="black",
                label=pretty_category_name(cat),
            )
            for cat in unique_cats
        ]
        + (
            [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    markersize=10,
                    markerfacecolor=GREY_CTRL,
                    markeredgecolor="black",
                    label="BLiMP-Control",
                )
            ]
            if "BLiMP-Control (Avg.)" in suite_rows
            else []
        )
        + [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=GREY_RAND,
                markeredgecolor="black",
                label="Random",
            ),
        ]
    )

    ax.legend(handles=handles, bbox_to_anchor=(1.2, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(
        f"{args.directory}/cross_validation_{args.dataset.lower()}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        f"{args.directory}/cross_validation_{args.dataset.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )
