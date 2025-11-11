import os
import csv

import pandas as pd


def convert(suite_name):
    df = pd.read_csv(f"raw/rublimp/{suite_name}.csv")

    with open(f"processed/rublimp/{suite_name}.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        max_words = 0
        all_good = []
        all_bad = []
        for i in range(len(df)):
            good_words = df.loc[i, "source_sentence"].split()
            bad_words = df.loc[i, "target_sentence"].split()
            max_words = max(max_words, max(len(good_words), len(bad_words)))
            all_good.append(good_words)
            all_bad.append(bad_words)
        header = ["stim" + str(i + 1) for i in range(max_words + 2)]
        writer.writerow(header)
        for i, (good_words, bad_words) in enumerate(zip(all_good, all_bad)):
            writer.writerow([2 * i + 1] + good_words + ["+"])
            writer.writerow([2 * i + 2] + bad_words + ["-"])


if __name__ == "__main__":
    for suite_name in os.listdir("raw/rublimp"):
        convert(suite_name[:-4])
