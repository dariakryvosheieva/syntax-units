#!/usr/bin/env python3
import argparse, glob, math, os, re, statistics, sys, csv
from collections import defaultdict

HEADER_RE = re.compile(r"^\D")         # non-digit line → suite header
ROW_RE    = re.compile(r"^\s*\d+")     # digit line → rank layer kind idx
MODEL_RE  = re.compile(r"finegrained_blimp_(.+?)_1%\.txt$")

def parse_layers_by_suite(path):
    """
    Parse one model file → (suite -> list[int layer]), plus all layer indices.
    """
    suite_layers = defaultdict(list)
    cur = None
    all_layers = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if HEADER_RE.match(line) and not ROW_RE.match(line):
                cur = line
                continue
            if not ROW_RE.match(line):
                continue
            parts = line.split()
            if len(parts) < 4:  # rank, layer, kind, idx
                continue
            try:
                layer = int(parts[1])
            except ValueError:
                continue
            if cur is None:
                cur = "UNKNOWN"
            suite_layers[cur].append(layer)
            all_layers.append(layer)

    return suite_layers, all_layers

def infer_indexing_and_N(all_layers):
    """
    Decide 0-based vs 1-based and infer N (layer count).
      - if a 0 exists → 0-based, N = max+1
      - elif 1 exists and no 0 → 1-based, N = max
      - else → assume 0-based, N = max+1
    """
    if not all_layers:
        return 0, 0, "unknown"

    if 0 in all_layers:
        one_based = False
        N = max(all_layers) + 1
        basis = "0-based"
    elif 1 in all_layers:
        one_based = True
        N = max(all_layers)
        basis = "1-based"
    else:
        one_based = False
        N = max(all_layers) + 1
        basis = "assumed 0-based"
    return (1 if one_based else 0), N, basis

def compute_threshold(N, one_based_flag, top_pct):
    """
    Given N, indexing basis, and percentage (0..100), return:
      top_count (number of top layers included) and threshold layer index.
    We clamp percentage into [0, 100] and allow top_count==0.
    """
    pct = max(0.0, min(100.0, float(top_pct)))
    top_count = int(math.ceil((pct / 100.0) * N))
    top_count = min(top_count, N)
    threshold = N - top_count + (1 if one_based_flag else 0)
    return top_count, threshold, pct

def model_name_from_path(p):
    m = MODEL_RE.search(os.path.basename(p))
    return m.group(1) if m else os.path.basename(p)

def parse_layers_override(s):
    """
    Parse a string like: gpt2-xl=48,llama-3-8b=32 → dict{name:int}
    """
    out = {}
    if not s:
        return out
    for pair in s.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            continue
        try:
            out[k] = int(v)
        except ValueError:
            pass
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Average (across models) % of top units in the last PCT% of layers, per test suite."
    )
    ap.add_argument("--dir", default="english/finegrained",
                    help="Directory containing finegrained_blimp_{model}_1%%.txt files.")
    ap.add_argument("--pattern", default="finegrained_blimp_*_1%.txt",
                    help="Glob pattern inside --dir.")
    ap.add_argument("--layers-override",
                    help="Optional per-model layer count, e.g.: 'gpt2-xl=48,llama-3-8b=32'")
    ap.add_argument("--top-pct", type=float, default=20.0,
                    help="Percent of *top layers* to treat as 'last %% of layers' (0..100). Default: 20.")
    ap.add_argument("--csv", metavar="OUT.csv", help="Also write results to CSV.")
    ap.add_argument("--show-model-stats", action="store_true",
                    help="Print inferred indexing and N per model.")
    ap.add_argument("--sort", choices=["suite","avg","n_models"], default="suite",
                    help="Sort output by suite name, average percent, or #models.")
    ap.add_argument("--desc", action="store_true", help="Sort descending.")
    args = ap.parse_args()

    pattern = os.path.join(args.dir, args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching: {pattern}", file=sys.stderr)
        sys.exit(1)

    layers_override = parse_layers_override(args.layers_override)

    # suite -> list of per-model percentages
    suite_to_model_percents = defaultdict(list)

    # For optional logging
    model_summaries = []

    for fp in files:
        model = model_name_from_path(fp)

        suite_layers, all_layers = parse_layers_by_suite(fp)
        if not all_layers:
            continue

        # Infer or override N
        one_based_flag, inferred_N, basis = infer_indexing_and_N(all_layers)
        N = layers_override.get(model, inferred_N)
        if N <= 0:
            continue

        top_count, threshold, pct_used = compute_threshold(N, one_based_flag, args.top_pct)

        if args.show_model_stats:
            model_summaries.append(
                (model, basis, N, top_count, threshold, len(suite_layers), len(all_layers))
            )

        # Compute per-suite percent for this model
        for suite, layers in suite_layers.items():
            total = len(layers)
            in_top_region = sum(1 for L in layers if L >= threshold)
            pct_in_region = (100.0 * in_top_region / total) if total else 0.0
            suite_to_model_percents[suite].append(pct_in_region)

    # Aggregate across models
    rows = []
    for suite, percs in suite_to_model_percents.items():
        avg = statistics.fmean(percs)
        sd = statistics.pstdev(percs) if len(percs) > 1 else 0.0
        rows.append({
            "suite": suite,
            "avg_percent_in_top_region": avg,
            "stddev": sd,
            "n_models": len(percs)
        })

    # Sort
    key = {
        "suite":    lambda r: r["suite"].lower(),
        "avg":      lambda r: r["avg_percent_in_top_region"],
        "n_models": lambda r: r["n_models"],
    }[args.sort]
    rows.sort(key=key, reverse=args.desc)

    # Optional per-model stats
    if args.show_model_stats and model_summaries:
        print("Per-model inference:")
        print(f"{'Model':30} {'Indexing':12} {'N':>4} {'Top#':>6} {'Thresh≥':>7} {'#Suites':>8} {'#Units':>8}")
        for m, basis, N, top_count, thr, n_suites, n_units in sorted(model_summaries):
            print(f"{m[:30]:30} {basis:12} {N:4d} {top_count:6d} {thr:7d} {n_suites:8d} {n_units:8d}")
        print("")

    # Print final table
    hdr_pct = max(0.0, min(100.0, float(args.top_pct)))
    print(f"Top region: last {hdr_pct}% of layers")
    print(f"{'Suite':50}  {'Avg % in top region':>21}  {'StdDev':>8}  {'#Models':>8}")
    print("-"*96)
    for r in rows:
        print(f"{r['suite'][:50]:50}  {r['avg_percent_in_top_region']:21.2f}  {r['stddev']:8.2f}  {r['n_models']:8d}")

    # CSV
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["suite","avg_percent_in_top_region","stddev","n_models"]
            )
            w.writeheader()
            w.writerows(rows)

if __name__ == "__main__":
    main()
