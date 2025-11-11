import csv

import pandas as pd

from local_datasets import process


df = pd.read_csv("raw/gulordava/generated.tab", sep="\t")
df = df.rename(str.strip, axis="columns")


with open(f"processed/gulordava/subject_verb_agreement_gulordava.csv", "w") as csv_file:
    writer = csv.writer(csv_file)

    max_words = 0

    all_good = []
    all_bad = []

    for i in range(0, len(df), 2):
        if df.iloc[i]["type"].strip() != "original":
            continue

        good_word = df.iloc[i]["form"].strip()
        bad_word = df.iloc[i + 1]["form"].strip()

        good_words = process(df.iloc[i]["sent"])
        bad_words = process(df.iloc[i + 1]["sent"], good_word, bad_word)

        max_words = max(max_words, max(len(good_words), len(bad_words)))

        all_good.append(good_words)
        all_bad.append(bad_words)

    header = ["stim" + str(i + 1) for i in range(max_words + 2)]
    writer.writerow(header)
    for i, (good_words, bad_words) in enumerate(zip(all_good, all_bad)):
        writer.writerow([2 * i + 1] + good_words + ["+"])
        writer.writerow([2 * i + 2] + bad_words + ["-"])
