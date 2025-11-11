import os
import json
import csv
import argparse


def convert(dataset, suite_name):
    with open(f"raw/{dataset}/{suite_name}.jsonl", "r") as json_file:
        json_list = list(json_file)

    with open(f"processed/{dataset}/{suite_name}.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        max_words = 0
        all_good = []
        all_bad = []
        for json_str in json_list:
            minimal_pair = json.loads(json_str)
            good_words = minimal_pair["sentence_good"].split()
            bad_words = minimal_pair["sentence_bad"].split()
            max_words = max(max_words, max(len(good_words), len(bad_words)))
            all_good.append(good_words)
            all_bad.append(bad_words)
        header = ["stim" + str(i + 1) for i in range(max_words + 2)]
        writer.writerow(header)
        for i, (good_words, bad_words) in enumerate(zip(all_good, all_bad)):
            writer.writerow([2 * i + 1] + good_words + ["+"])
            writer.writerow([2 * i + 2] + bad_words + ["-"])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, required=True)
    args = argparser.parse_args()
    dataset = args.dataset

    for suite_name in os.listdir(f"raw/{dataset}"):
        convert(dataset, suite_name[:-6])
