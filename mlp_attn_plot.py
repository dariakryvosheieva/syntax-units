import glob
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


Unit = Tuple[int, int, str, int]


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


def compute_proportions(phenomena: Dict[str, List[Unit]]):
    rows = []
    for ph, items in phenomena.items():
        mlp_count = sum(1 for *_, kind, __ in items if kind == "mlp")
        attn_count = sum(1 for *_, kind, __ in items if kind == "attn")
        total = mlp_count + attn_count
        if total == 0:
            mlp_prop = 0.0
            attn_prop = 0.0
        else:
            mlp_prop = mlp_count / total
            attn_prop = attn_count / total
        rows.append(
            {
                "phenomenon": ph,
                "total_units": total,
                "mlp_count": mlp_count,
                "attn_count": attn_count,
                "mlp_prop": mlp_prop,
                "attn_prop": attn_prop,
            }
        )
    return rows


def summarize_file(path: str) -> Dict[str, float]:
    ph = parse_units_file(path)
    rows = compute_proportions(ph)
    return {r["phenomenon"]: r["mlp_prop"] for r in rows}


def aggregate_models(file_paths: List[str]):
    model_names = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
    per_model_maps = [summarize_file(p) for p in file_paths]

    all_phenomena = set()
    for m in per_model_maps:
        all_phenomena.update(m.keys())
    all_phenomena = list(all_phenomena)

    rows = []
    for ph in all_phenomena:
        values = []
        for m in per_model_maps:
            values.append(m.get(ph, np.nan))
        values_arr = np.array(values, dtype=float)

        if np.isnan(values_arr).all():
            avg_mlp = 0.0
        else:
            avg_mlp = float(np.nanmean(values_arr))
        rows.append(
            {
                "phenomenon": ph,
                "avg_mlp_prop": avg_mlp,
                "avg_attn_prop": 1.0 - avg_mlp,
                "model_mlp_props": values_arr,
            }
        )
    return rows, model_names


def plot_avg_with_model_dots(
    rows,
    model_names: List[str],
):
    rows.sort(key=lambda r: r["avg_mlp_prop"], reverse=True)

    n = len(rows)
    fig_h = max(5.0, 0.28 * n + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y = list(range(n))
    avg_mlp = [r["avg_mlp_prop"] for r in rows]
    avg_attn = [r["avg_attn_prop"] for r in rows]
    labels = [r["phenomenon"] for r in rows]

    mlp_color = "#2f8bcc"
    attn_color = "#ff7dbe"

    ax.barh(y, avg_mlp, color=mlp_color, height=1.0, edgecolor="black", label="MLP")
    ax.barh(
        y,
        avg_attn,
        left=avg_mlp,
        color=attn_color,
        height=1.0,
        edgecolor="black",
        label="Attention",
    )

    n_models = len(model_names)
    if n_models > 0:
        jitter = np.linspace(-0.18, 0.18, n_models)
        for j in range(n_models):
            xs = []
            ys = []
            for i, r in enumerate(rows):
                v = r["model_mlp_props"][j] if j < len(r["model_mlp_props"]) else np.nan
                if not np.isnan(v):
                    xs.append(float(v))
                    ys.append(i + jitter[j])
            if xs:
                ax.scatter(xs, ys, s=18, color="black", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_xlabel("MLP Proportion (%)", fontsize=14)

    ax.set_title("MLP/Attention Breakdown of Responsive Units", fontsize=16)

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=12)

    ax.legend(loc="lower right", fontsize=14)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()

    fig.savefig("english/finegrained/mlp_attn.png")
    fig.savefig("english/finegrained/mlp_attn.pdf")


def main():
    file_paths = sorted(glob.glob("english/finegrained/*.txt"))

    rows, model_names = aggregate_models(file_paths)

    plot_avg_with_model_dots(
        rows,
        model_names,
    )


if __name__ == "__main__":
    main()
