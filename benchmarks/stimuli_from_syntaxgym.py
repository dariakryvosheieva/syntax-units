import os
import json
import csv


def format_list(r):
    l = []
    for region_idx, region in enumerate(r["regions"]):
        l += region["content"].split()
    l[-1] += "."
    return l


def write_suite(filename, max_words, all_good, all_bad):
    with open(f"processed/syntaxgym/{filename}.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        header = ["stim" + str(i + 1) for i in range(max_words + 2)]
        writer.writerow(header)
        for i, (good_words, bad_words) in enumerate(zip(all_good, all_bad)):
            writer.writerow([2 * i + 1] + good_words + ["+"])
            writer.writerow([2 * i + 2] + bad_words + ["-"])


def convert(suite_name):
    with open(f"raw/syntaxgym/{suite_name}.json", "r") as json_file:
        json_dict = json.load(json_file)

    if suite_name.startswith("filler_gap"):
        max_words_with_gap, max_words_no_gap = 0, 0
        with_gap_all_good, with_gap_all_bad, no_gap_all_good, no_gap_all_bad = (
            [],
            [],
            [],
            [],
        )

        for item_idx, item in enumerate(json_dict["items"]):
            with_gap_good = item["conditions"][2]
            with_gap_bad = item["conditions"][0]
            no_gap_good = item["conditions"][1]
            no_gap_bad = item["conditions"][3]

            with_gap_good_words = format_list(with_gap_good)
            with_gap_all_good.append(with_gap_good_words)
            max_words_with_gap = max(max_words_with_gap, len(with_gap_good_words))

            with_gap_bad_words = format_list(with_gap_bad)
            with_gap_all_bad.append(with_gap_bad_words)
            max_words_with_gap = max(max_words_with_gap, len(with_gap_bad_words))

            no_gap_good_words = format_list(no_gap_good)
            no_gap_all_good.append(no_gap_good_words)
            max_words_no_gap = max(max_words_no_gap, len(no_gap_good_words))

            no_gap_bad_words = format_list(no_gap_bad)
            no_gap_all_bad.append(no_gap_bad_words)
            max_words_no_gap = max(max_words_no_gap, len(no_gap_bad_words))

        write_suite(
            f"{suite_name}_with_gap",
            max_words_with_gap,
            with_gap_all_good,
            with_gap_all_bad,
        )
        write_suite(
            f"{suite_name}_no_gap", max_words_no_gap, no_gap_all_good, no_gap_all_bad
        )

    if suite_name.startswith("subject_verb_agreement"):
        max_words_sg, max_words_pl = 0, 0
        sg_all_good, sg_all_bad, pl_all_good, pl_all_bad = [], [], [], []

        for item_idx, item in enumerate(json_dict["items"]):
            sg_good = item["conditions"][1]
            sg_bad = item["conditions"][3]
            pl_good = item["conditions"][0]
            pl_bad = item["conditions"][2]

            sg_good_words = format_list(sg_good)
            sg_all_good.append(sg_good_words)
            max_words_sg = max(max_words_sg, len(sg_good_words))

            sg_bad_words = format_list(sg_bad)
            sg_all_bad.append(sg_bad_words)
            max_words_sg = max(max_words_sg, len(sg_bad_words))

            pl_good_words = format_list(pl_good)
            pl_all_good.append(pl_good_words)
            max_words_pl = max(max_words_pl, len(pl_good_words))

            pl_bad_words = format_list(pl_bad)
            pl_all_bad.append(pl_bad_words)
            max_words_pl = max(max_words_pl, len(pl_bad_words))

        write_suite(
            suite_name,
            max(max_words_sg, max_words_pl),
            sg_all_good + pl_all_good,
            sg_all_bad + pl_all_bad,
        )


if __name__ == "__main__":
    for suite_name in os.listdir("raw/syntaxgym"):
        convert(suite_name[:-5])
