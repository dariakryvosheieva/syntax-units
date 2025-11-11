import argparse
import sys

import pandas as pd
import numpy as np
from scipy import stats


def one_sample_test(vals: np.ndarray, baseline: float) -> tuple[float, float]:
    t, p = stats.ttest_1samp(vals, popmean=baseline, alternative="greater")
    return float(t), float(p)


def two_sample_welch_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    t, p = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
    return float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--mode", choices=["one", "two"], required=True)
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--level", default=None)
    ap.add_argument("--x-level", default=None)
    ap.add_argument("--y-level", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    perc_col = "perc_overlap"
    pair_col = "pair_type"

    if args.mode == "one":
        sample_df = df
        if args.level is not None:
            sample_df = df[df[pair_col].astype(str) == str(args.level)]

        vals = pd.to_numeric(sample_df[perc_col], errors="coerce").dropna().to_numpy()

        t, p = one_sample_test(vals, args.baseline)
        label = f"level='{args.level}'" if args.level is not None else "all rows"
        print(
            f"one-sample t (greater; {label}; baseline={args.baseline}): t={t:.6g}, p={p:.6g}"
        )
        return

    pair_series = df[pair_col].astype(str)
    X = (
        pd.to_numeric(
            df.loc[pair_series == str(args.x_level), perc_col], errors="coerce"
        )
        .dropna()
        .to_numpy()
    )
    Y = (
        pd.to_numeric(
            df.loc[pair_series == str(args.y_level), perc_col], errors="coerce"
        )
        .dropna()
        .to_numpy()
    )

    t, p = two_sample_welch_test(X, Y)
    print(
        f"two-sample Welch t (X>Y; X='{args.x_level}', Y='{args.y_level}'): t={t:.6g}, p={p:.6g}"
    )


if __name__ == "__main__":
    main()
