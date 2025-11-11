import argparse, csv, glob, os, random


def read_content_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if first:
                first = False
                continue
            rows.append([c.strip() for c in row])
    return rows


def chunk_pairs(rows):
    return [[rows[i], rows[i+1]] for i in range(0, len(rows), 2)]


def write_csv(path, header_len, rows):
    header = [f"stim{i}" for i in range(1, header_len + 1)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_dir", type=str)
    argparser.add_argument("--seed", type=int, default=42)
    args = argparser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    pos = os.path.basename(args.input_dir).split("-")[1]

    all_pairs, all_rows_flat = [], []
    for p in csv_files:
        rows = read_content_rows(p)
        pairs = chunk_pairs(rows)
        all_pairs.extend(pairs)
        all_rows_flat.extend([r for pair in pairs for r in pair])

    max_len = max(len(r) for r in all_rows_flat)

    rng = random.Random(args.seed)
    shuffled_pairs = all_pairs[:]
    rng.shuffle(shuffled_pairs)
    shuffled_flat = [r for pair in shuffled_pairs for r in pair]
    write_csv(f"blimp-{pos}-sub-concat/blimp_{pos}_sub.csv", max_len, shuffled_flat)


if __name__ == "__main__":
    main()
