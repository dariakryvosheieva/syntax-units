import os
import argparse

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from categories import CATEGORIES
from model_utils import get_num_blocks, get_hidden_dim
from plot_utils import build_model_style_maps, add_model_scatter
from utils import cross_validation, cross_overlap, new_matrix, random_overlap_expected


BLIMP = CATEGORIES["blimp"]

CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")

MODELS = [
    "openai-community/gpt2-xl",
    "meta-llama/Llama-3.2-3B",
    "tiiuae/Falcon3-3B-Base",
    "microsoft/Phi-4-mini-instruct",
    "google/gemma-3-4b-pt",
    "deepseek-ai/deepseek-llm-7b-base",
    "mistralai/Mistral-7B-v0.3",
]


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_networks(dataset, category):
    return [
        suite[:-4]
        for suite in os.listdir(f"benchmarks/processed/{dataset}")
        if suite[:-4] in BLIMP[category] or suite.startswith(category)
    ]


def get_cross_overlap(
    model_name, model, tokenizer, dataset, category, pooling, percentage
):
    fname = f"english/generalization/generalization_blimp_{dataset}_{model_name}_{category}.csv"

    if os.path.exists(fname):
        cross_matrix = pd.read_csv(fname, index_col="Unnamed: 0")
    else:
        blimp_networks = get_networks("blimp", category)
        other_networks = get_networks(dataset, category)

        cross_matrix = new_matrix(blimp_networks, other_networks)

        for blimp_network in blimp_networks:
            for other_network in other_networks:
                overlap = np.sum(
                    cross_overlap(
                        model_name,
                        model,
                        tokenizer,
                        "blimp",
                        dataset,
                        blimp_network,
                        other_network,
                        pooling,
                        device,
                        percentage,
                    )
                )
                cross_matrix.loc[blimp_network, other_network] = overlap

        cross_matrix.to_csv(fname)

    return cross_matrix.mean().mean()


def get_inner_overlap(
    model_name, model, tokenizer, dataset, category, pooling, percentage
):
    networks = get_networks(dataset, category)

    if len(networks) == 1:
        fname = f"english/generalization/cross-validation_{dataset}_{model_name}_{percentage}%_2-fold.txt"

        if os.path.exists(fname):
            with open(fname, "r") as f:
                lines = f.readlines()
                for line in lines:
                    num_units, pct_units, suite_name = line.split()
                    if suite_name == networks[0]:
                        return int(num_units)

        score = np.sum(
            cross_validation(
                model_name,
                model,
                tokenizer,
                dataset,
                networks[0],
                num_folds=2,
                pooling=pooling,
                device=device,
                percentage=percentage,
            )
        )
        with open(fname, "w") as f:
            num_units = get_num_blocks(model_name) * get_hidden_dim(model_name)
            percentage_units = int(num_units * percentage / 100)
            f.write(f"{score} ({score / percentage_units * 100:.2f}%) {networks[0]}")
        return score
    else:
        fname = f"english/cross-overlap/cross-overlap_{dataset}_{dataset}_{model_name}_{percentage}%.csv"

        if os.path.exists(fname):
            matrix = pd.read_csv(fname, index_col="Unnamed: 0")
        else:
            matrix = new_matrix(networks, networks)
            for network_1 in networks:
                for network_2 in networks:
                    overlap = np.sum(
                        cross_overlap(
                            model_name,
                            model,
                            tokenizer,
                            dataset,
                            dataset,
                            network_1,
                            network_2,
                            pooling,
                            device,
                            percentage,
                        )
                    )
                    matrix.loc[network_1, network_2] = overlap

        return matrix.mean().mean()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--percentage", type=float, default=1)
    argparser.add_argument("--seed", type=int, default=12)
    argparser.add_argument(
        "--pooling", type=str, default="last-token", choices=["last-token", "mean"]
    )

    args = argparser.parse_args()

    percentage = args.percentage
    pooling = args.pooling

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model_basenames = [os.path.basename(m) for m in MODELS]
    model_list, model_to_color, model_to_marker = build_model_style_maps(
        model_basenames
    )

    svagr = "subject_verb_agreement"
    fg = "filler_gap"

    percentage_units, rand_overlap = {}, {}
    for model_name in model_basenames:
        num_units = get_num_blocks(model_name) * get_hidden_dim(model_name)
        percentage_units[model_name] = int(num_units * percentage / 100)
        rand_overlap[model_name] = random_overlap_expected(
            num_units, percentage_units[model_name]
        )

    (
        mean_cross_overlap_syntaxgym,
        mean_cross_overlap_gulordava,
        mean_cross_overlap_linzen,
    ) = ({}, {}, {})
    (
        mean_blimp_overlap,
        mean_syntaxgym_overlap,
        mean_gulordava_overlap,
        mean_linzen_overlap,
    ) = ({}, {}, {}, {})

    mean_cross_overlap_fg, mean_blimp_overlap_fg, mean_syntaxgym_overlap_fg = {}, {}, {}

    for model_name in MODELS:
        print(f"> Running with model {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_name = os.path.basename(model_name)

        mean_cross_overlap_syntaxgym[model_name] = get_cross_overlap(
            model_name, model, tokenizer, "syntaxgym", svagr, pooling, percentage
        )
        mean_cross_overlap_gulordava[model_name] = get_cross_overlap(
            model_name, model, tokenizer, "gulordava", svagr, pooling, percentage
        )
        mean_cross_overlap_linzen[model_name] = get_cross_overlap(
            model_name, model, tokenizer, "linzen", svagr, pooling, percentage
        )

        mean_blimp_overlap[model_name] = get_inner_overlap(
            model_name, model, tokenizer, "blimp", svagr, pooling, percentage
        )
        mean_syntaxgym_overlap[model_name] = get_inner_overlap(
            model_name, model, tokenizer, "syntaxgym", svagr, pooling, percentage
        )
        mean_gulordava_overlap[model_name] = get_inner_overlap(
            model_name, model, tokenizer, "gulordava", svagr, pooling, percentage
        )
        mean_linzen_overlap[model_name] = get_inner_overlap(
            model_name, model, tokenizer, "linzen", svagr, pooling, percentage
        )

        mean_cross_overlap_fg[model_name] = get_cross_overlap(
            model_name, model, tokenizer, "syntaxgym", fg, pooling, percentage
        )
        mean_blimp_overlap_fg[model_name] = get_inner_overlap(
            model_name, model, tokenizer, "blimp", fg, pooling, percentage
        )
        mean_syntaxgym_overlap_fg[model_name] = get_inner_overlap(
            model_name, model, tokenizer, "syntaxgym", fg, pooling, percentage
        )

    def avg_absolute(d):
        return sum(d.values()) / len(d)

    def avg_percent(d):
        return sum(d[m] / percentage_units[m] * 100 for m in d) / len(d)

    with PdfPages("english/generalization/generalization.pdf") as pdf:
        y_max = sum(percentage_units.values()) / len(percentage_units)

        fig, (ax0, ax1) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 15),
            sharey=False,
        )

        across_row_labels = [
            "BLiMP ×\nSyntaxGym",
            "BLiMP ×\nGulordava",
            "BLiMP ×\nLinzen",
        ]
        across_row_series = [
            avg_absolute(mean_cross_overlap_syntaxgym),
            avg_absolute(mean_cross_overlap_gulordava),
            avg_absolute(mean_cross_overlap_linzen),
        ]
        across_row_percentages = [
            avg_percent(mean_cross_overlap_syntaxgym),
            avg_percent(mean_cross_overlap_gulordava),
            avg_percent(mean_cross_overlap_linzen),
        ]
        x_across = np.arange(len(across_row_labels))

        bars_across = ax0.bar(
            x_across, across_row_series, width=0.8, color="#FFB668", edgecolor="black", linewidth=0.5
        )

        add_model_scatter(
            mean_cross_overlap_syntaxgym,
            x_across[0],
            ax0,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )
        add_model_scatter(
            mean_cross_overlap_gulordava,
            x_across[1],
            ax0,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )
        add_model_scatter(
            mean_cross_overlap_linzen,
            x_across[2],
            ax0,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )

        for bar, pct in zip(bars_across, across_row_percentages):
            ax0.annotate(
                f"{pct:.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                zorder=9,
                fontsize=15,
            )

        within_row_labels = ["BLiMP", "SyntaxGym", "Gulordava", "Linzen"]
        within_row_series = [
            avg_absolute(mean_blimp_overlap),
            avg_absolute(mean_syntaxgym_overlap),
            avg_absolute(mean_gulordava_overlap),
            avg_absolute(mean_linzen_overlap),
        ]
        within_row_percentages = [
            avg_percent(mean_blimp_overlap),
            avg_percent(mean_syntaxgym_overlap),
            avg_percent(mean_gulordava_overlap),
            avg_percent(mean_linzen_overlap),
        ]
        gap = 0.3
        x_within_start = len(across_row_labels) + gap
        x_within = np.arange(len(within_row_labels)) + x_within_start

        bars_within = ax0.bar(
            x_within, within_row_series, width=0.8, color="#89CCF1", edgecolor="black", linewidth=0.5
        )

        add_model_scatter(
            mean_blimp_overlap,
            x_within[0],
            ax0,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )
        add_model_scatter(
            mean_syntaxgym_overlap,
            x_within[1],
            ax0,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )
        add_model_scatter(
            mean_gulordava_overlap,
            x_within[2],
            ax0,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )
        add_model_scatter(
            mean_linzen_overlap,
            x_within[3],
            ax0,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )

        for bar, pct in zip(bars_within, within_row_percentages):
            ax0.annotate(
                f"{pct:.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                zorder=9,
                fontsize=15,
            )

        random_label = ["Random"]
        random_series = [avg_absolute(rand_overlap)]
        random_pct = avg_percent(rand_overlap)

        x_random = x_within[-1] + gap + 1
        bar_rand = ax0.bar(
            [x_random], random_series, width=0.8, color="#C0C0C0", edgecolor="black", linewidth=0.5
        )

        ax0.annotate(
            f"{random_pct:.1f}%",
            (x_random, random_series[0]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            zorder=9,
            fontsize=15,
        )

        centers_across = [b.get_x() + b.get_width() / 2 for b in bars_across]
        centers_within = [b.get_x() + b.get_width() / 2 for b in bars_within]
        center_random = bar_rand[0].get_x() + bar_rand[0].get_width() / 2

        ax0.set_xticks(
            np.concatenate([centers_across, centers_within, [center_random]])
        )
        ax0.set_xticklabels(
            across_row_labels + within_row_labels + random_label,
            ha="center",
            fontsize=11,
        )
        ax0.set_title("S-V agreement", fontsize=19, pad=12)

        fg_labels = ["BLiMP ×\nSyntaxGym", "BLiMP", "SyntaxGym", "Random"]
        fg_series = [
            avg_absolute(mean_cross_overlap_fg),
            avg_absolute(mean_blimp_overlap_fg),
            avg_absolute(mean_syntaxgym_overlap_fg),
            avg_absolute(rand_overlap),
        ]
        fg_percentages = [
            avg_percent(mean_cross_overlap_fg),
            avg_percent(mean_blimp_overlap_fg),
            avg_percent(mean_syntaxgym_overlap_fg),
            avg_percent(rand_overlap),
        ]
        x_fg = np.arange(len(fg_labels))

        fg_colors = ["#FFB668", "#89CCF1", "#89CCF1", "#C0C0C0"]

        bars_fg = ax1.bar(
            x_fg, fg_series, width=0.8, color=fg_colors, edgecolor="black"
        )

        add_model_scatter(
            mean_cross_overlap_fg,
            x_fg[0],
            ax1,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )
        add_model_scatter(
            mean_blimp_overlap_fg,
            x_fg[1],
            ax1,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )
        add_model_scatter(
            mean_syntaxgym_overlap_fg,
            x_fg[2],
            ax1,
            model_to_color,
            model_to_marker,
            rng,
            s=50,
            jitter=0.15,
            vertical=True,
        )

        for bar, pct in zip(bars_fg, fg_percentages):
            ax1.annotate(
                f"{pct:.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                zorder=9,
                fontsize=15,
            )

        c_fg = [b.get_x() + b.get_width() / 2 for b in bars_fg]
        ax1.set_xticks(c_fg)
        ax1.set_xticklabels(fg_labels, ha="center", fontsize=12)
        ax1.set_title("Filler-gap", fontsize=19, pad=12)

        for ax in (ax0, ax1):
            ax.tick_params(axis="y", labelsize=15)
            ax.set_ylabel("Number of Units", fontsize=17)
            ax.spines[["top", "right"]].set_visible(False)
        ax0.set_ylim(0, y_max)
        ax1.set_ylim(0, 600)

        handles = [
            Line2D(
                [0],
                [0],
                marker=model_to_marker[m],
                color="w",
                markerfacecolor=model_to_color[m],
                markeredgecolor="black",
                markeredgewidth=0.2,
                markersize=9,
                label=m,
            )
            for m in sorted(model_basenames)
        ]

        fig.legend(
            handles=handles,
            title="Models",
            frameon=True,
            loc="center left",
            ncol=2,
            fontsize=15,
            title_fontsize=17,
            bbox_to_anchor=(0.12, 0.42),
        )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
