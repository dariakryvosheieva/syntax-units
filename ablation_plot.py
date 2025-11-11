import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from categories import CATEGORIES
from plot_utils import pretty_category_name, CATEGORY_COLORS


matplotlib.rcParams.update(
    {
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.6,
        "lines.solid_capstyle": "round",
    }
)


def bootstrap_ci(values, iters=10000, alpha=0.05, rng=None):
    v = np.asarray(values, dtype=float)
    if rng is None:
        rng = np.random.default_rng(12345)
    if v.size == 1:
        return 0.0, 0.0
    means = rng.choice(v, size=(iters, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha)])
    mu = v.mean()
    return mu - lo, hi - mu


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="BLiMP")
    p.add_argument("--directory", default=f"english/ablation")
    p.add_argument("--title", default=f"Average ablation effect")
    p.add_argument(
        "--display", default="error-bars", choices=["error-bars", "model-markers"]
    )
    args = p.parse_args()

    original, top, random = {}, {}, {}
    model_names = set()

    for file in os.listdir(args.directory):
        if file.startswith(f"ablation_{args.dataset.lower()}") and file.endswith(
            "%.txt"
        ):
            model_name = file.split("_")[2]
            model_names.add(model_name)

            print(file)

            with open(os.path.join(args.directory, file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    suite, nums = parts
                    suite = suite[:-1]
                    o, t, r = map(float, nums.split("/"))

                    original.setdefault(suite, {})[model_name] = o
                    top.setdefault(suite, {})[model_name] = t
                    random.setdefault(suite, {})[model_name] = r

    num_top = 0
    sum_top = 0
    
    num_random = 0
    sum_random = 0

    results = []
    for suite in original:
        o_vals = list(original[suite].values())
        t_vals = list(top[suite].values())
        r_vals = list(random[suite].values())

        for m in model_names:
            sum_top += top[suite][m] - original[suite][m]
            num_top += 1
            
            sum_random += random[suite][m] - original[suite][m]
            num_random += 1

        results.append(
            {
                "Suite": suite,
                "Original": np.mean(o_vals),
                "Top": np.mean(t_vals),
                "Random": np.mean(r_vals),
            }
        )

    print("Average top ablation effect: ", sum_top / num_top)
    print("Average random ablation effect: ", sum_random / num_random)

    df = pd.DataFrame(results)
    df["Top - Original"] = df["Top"] - df["Original"]
    df["Random - Original"] = df["Random"] - df["Original"]
    df = df.sort_values("Top - Original").reset_index(drop=True)

    top_ci_low, top_ci_high = [], []
    rand_ci_low, rand_ci_high = [], []

    for suite in df["Suite"]:
        top_vals = [
            top[suite][m] - original[suite][m] for m in model_names if m in top[suite]
        ]
        rand_vals = [
            random[suite][m] - original[suite][m]
            for m in model_names
            if m in random[suite]
        ]

        cl, ch = bootstrap_ci(top_vals)
        top_ci_low.append(cl)
        top_ci_high.append(ch)

        cl, ch = bootstrap_ci(rand_vals)
        rand_ci_low.append(cl)
        rand_ci_high.append(ch)

    spacing = 1.5
    bar_h = 0.5
    df["Position"] = np.arange(len(df)) * spacing

    fig_h = max(6, 0.3 + len(df) * 0.45)
    fig, ax = plt.subplots(figsize=(15, fig_h))

    edge_lw = 0.35
    err_kw = dict(elinewidth=0.5, capthick=0.5, capsize=3, ecolor="0.3")

    category_map = {
        suite_name: cat
        for cat, subdict in CATEGORIES["blimp"].items()
        for suite_name in subdict
    }

    GREY_RAND = "#888888"

    unique_cats = sorted({category_map[s] for s in df["Suite"] if s in category_map})

    cat2color = {
        cat: CATEGORY_COLORS[args.dataset.lower()][i]
        for i, cat in enumerate(unique_cats)
    }

    top_bar_colors = [cat2color[category_map[suite]] for suite in df["Suite"]]
    random_bar_colors = [GREY_RAND] * len(df)

    ax.barh(
        df["Position"] + bar_h / 2,
        df["Top - Original"],
        height=bar_h,
        color=top_bar_colors,
        edgecolor="black",
        linewidth=edge_lw,
        antialiased=True,
        xerr=([top_ci_low, top_ci_high] if args.display == "error-bars" else None),
        error_kw=err_kw,
    )

    ax.barh(
        df["Position"] - bar_h / 2,
        df["Random - Original"],
        height=bar_h,
        color=random_bar_colors,
        edgecolor="black",
        linewidth=edge_lw,
        antialiased=True,
        xerr=([rand_ci_low, rand_ci_high] if args.display == "error-bars" else None),
        error_kw=err_kw,
    )

    model_list = sorted(model_names)
    y_pos = df["Position"].values

    if args.display == "model-markers":
        for i, suite in enumerate(df["Suite"]):
            if suite in top:
                for j, model in enumerate(model_list):
                    if model not in top[suite]:
                        continue
                    score = top[suite][model] - original[suite][model]
                    ax.scatter(
                        score,
                        y_pos[i] + bar_h / 2,
                        color="black",
                        marker="o",
                        s=10,
                        zorder=5,
                    )

            if suite in random:
                for j, model in enumerate(model_list):
                    if model not in random[suite]:
                        continue
                    score_r = random[suite][model] - original[suite][model]
                    ax.scatter(
                        score_r,
                        y_pos[i] - bar_h / 2,
                        color="black",
                        marker="o",
                        s=10,
                        zorder=5,
                    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            linewidth=edge_lw,
            markersize=12,
            markerfacecolor=cat2color[cat],
            markeredgecolor="black",
            label=pretty_category_name(cat),
        )
        for cat in unique_cats
    ] + [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            linewidth=edge_lw,
            markersize=12,
            markerfacecolor=GREY_RAND,
            markeredgecolor="black",
            label="Random",
        ),
    ]

    ax.legend(handles=handles, loc="center left", frameon=True, fontsize=20)

    ax.invert_yaxis()
    ax.set_yticks([])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, which="both", color="gray", linestyle="-", alpha=0.2)

    ax.xaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1, decimals=0, symbol=None)
    )
    ax.set_xlabel("Accuracy Difference (%)", fontsize=24)
    ax.tick_params(axis="x", labelsize=24)
    ax.set_title(args.title, fontsize=27)

    ymin = df["Position"].min() - bar_h
    ymax = df["Position"].max() + bar_h
    ax.set_ylim(ymax, ymin)

    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(
        f"{args.directory}/ablation_{args.dataset.lower()}_{args.display}.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        f"{args.directory}/ablation_{args.dataset.lower()}_{args.display}.jpg", dpi=300
    )
