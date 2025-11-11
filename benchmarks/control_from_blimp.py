import os
import csv
import argparse

import pandas as pd


def convert(dataset, suite_name):
    df = pd.read_csv(f"processed/{dataset}/{suite_name}.csv")

    with open(f"processed/{dataset}-control/{suite_name}.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        max_words = 0
        for i in range(0, df.shape[0], 2):
            count = 0
            for j in range(df.shape[1]):
                count += 1
                if df.iloc[i, j] == "+":
                    if i % 4 != 0:
                        df.iloc[i, j] = "-"
                    break
            max_words = max(max_words, count)
        header = ["stim" + str(i + 1) for i in range(max_words + 2)]
        writer.writerow(header)
        for i in range(0, len(df), 2):
            row = []
            for j in range(df.shape[1]):
                row.append(df.iloc[i, j])
                if df.iloc[i, j] in ["+", "-"]:
                    break
            writer.writerow(row)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, required=True)
    args = argparser.parse_args()
    dataset = args.dataset

    for suite_name in os.listdir(f"processed/{dataset}"):
        convert(dataset, suite_name[:-4])
