from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from model_utils import get_num_blocks, get_hidden_dim

dist_path = Path("multilingual/multiblimp/language_pair_distances.csv")
overlap_path = Path("multilingual/multiblimp/cross-overlap_multiblimp_SV-#_gemma-3-4b-pt_1.0%.csv")
out_path = "multilingual/multiblimp/overlap_vs_syntactic_distance"

dist_df = pd.read_csv(dist_path)

overlap_df = pd.read_csv(overlap_path, index_col=0)

syn_col = "syntactic_dist"
lang1_col, lang2_col = "l1", "l2"

def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))

dist_small = (
    dist_df[[lang1_col, lang2_col, syn_col, "has_missing_syntax"]]
    .dropna()
    .query("has_missing_syntax == False")
    .query(f"{lang1_col} != {lang2_col}")         # <- exclude self distances
    .assign(pair=lambda d: d.apply(lambda r: pair_key(r[lang1_col], r[lang2_col]), axis=1))
    .drop_duplicates(subset=["pair"])              # <- enforce 1 row per pair
    .reset_index(drop=True)
)

langs = list(overlap_df.index)

pairs = []
for i, a in enumerate(langs):
    for j, b in enumerate(langs):
        if j <= i:
            continue
        val = overlap_df.loc[a, b]
        pairs.append((a, b, float(val)))
        
overlap_long = pd.DataFrame(pairs, columns=["lang_a", "lang_b", "overlap"])
overlap_long = overlap_long[overlap_long["lang_a"] != overlap_long["lang_b"]].copy()
overlap_long["pair"] = overlap_long.apply(lambda r: pair_key(r["lang_a"], r["lang_b"]), axis=1)
overlap_long = overlap_long.drop_duplicates(subset=["pair"]).reset_index(drop=True)

merged = overlap_long.merge(
    dist_small[["pair", syn_col]],
    on="pair",
    how="inner",
).rename(columns={syn_col: "syntactic_distance"})

mask = ~np.isclose(merged["syntactic_distance"].to_numpy(), 0.0, atol=1e-12)
merged = merged.loc[mask].reset_index(drop=True)

MODEL_NAME = "gemma-3-4b-pt"
TARGET = 1.0
TARGET_SET_SIZE = 0.01 * get_num_blocks(MODEL_NAME) * get_hidden_dim(MODEL_NAME)

merged["overlap_pct"] = merged["overlap"] / TARGET_SET_SIZE * 100

remaining_langs = pd.unique(merged[['lang_a', 'lang_b']].values.ravel())
num_remaining = remaining_langs.size
print(f"Number of unique languages: {num_remaining}")

fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
ax.scatter(
    1 - merged["syntactic_distance"],
    merged["overlap_pct"],
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5,
    marker="o",
    color="blue",
    s=10,
)

ax.set_xlabel("Syntactic similarity", fontsize=12)
ax.set_ylabel("Percentage of units", fontsize=12)

from scipy.stats import spearmanr

x = 1 - merged["syntactic_distance"].to_numpy()
y = merged["overlap_pct"].to_numpy()
n = len(merged)

slope, intercept, r, p_lin, stderr = linregress(x, y)
xline = np.linspace(x.min(), x.max(), 200)
yline = slope * xline + intercept
ax.plot(xline, yline, linewidth=1, linestyle="--", color="darkblue", zorder=0)

spearman = spearmanr(x, y, nan_policy="omit")
rho, pval = float(spearman.correlation), float(spearman.pvalue)

title = "Overlap vs. syntactic similarity (Gemma, MultiBLiMP SV-#)"
if np.isfinite(rho):
    title += f"\nSpearman ρ={rho:.3f} (p={pval:.3g}); n={n}"
ax.set_title(title, pad=10, fontsize=14)

ax.set_ylim(-5, 80)

ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
fig.tight_layout()
fig.savefig(out_path + ".pdf")
fig.savefig(out_path + ".png")
