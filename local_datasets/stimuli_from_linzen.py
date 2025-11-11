import csv

import pandas as pd

from local_datasets import process


if __name__ == "__main__":
    df = pd.read_csv(f"raw/linzen/lgd_dataset.tsv", sep="\t")

    max_words = 0

    all_good = []
    all_bad = []

    for i in range(0, len(df), 2):
        bad_word = df.iloc[i, 4]

        good_words = process(df.iloc[i, 1])
        bad_words = process(df.iloc[i, 2], "***mask***", bad_word)

        max_words = max(max_words, max(len(good_words), len(bad_words)))

        all_good.append(good_words)
        all_bad.append(bad_words)

    header = ["stim" + str(i + 1) for i in range(max_words + 2)]

    with open(f"processed/linzen/subject_verb_agreement_linzen.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for i, (good_words, bad_words) in enumerate(zip(all_good, all_bad)):
            writer.writerow([2 * i + 1] + good_words + ["+"])
            writer.writerow([2 * i + 2] + bad_words + ["-"])
