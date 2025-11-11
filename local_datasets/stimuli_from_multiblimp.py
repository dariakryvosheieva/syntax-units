import os
import csv
from datasets import get_dataset_config_names, load_dataset

languages = get_dataset_config_names("jumelet/multiblimp")
phenomena = ["SV-P", "SV-#", "SV-G", "SP-P", "SP-#", "SP-G"]

for lang in languages:
    ds = load_dataset("jumelet/multiblimp", lang)

    os.makedirs(f"processed/multiblimp/{lang}", exist_ok=True)

    files = {
        phenomenon: open(f"processed/multiblimp/{lang}/{phenomenon}.csv", "w")
        for phenomenon in phenomena
    }

    writers = {phenomenon: csv.writer(files[phenomenon]) for phenomenon in phenomena}

    max_words = {phenomenon: 0 for phenomenon in phenomena}

    all_good = {phenomenon: [] for phenomenon in phenomena}
    all_bad = {phenomenon: [] for phenomenon in phenomena}

    for row in ds["train"]:
        phenomenon = row["phenomenon"]

        good_words = row["sen"].split()
        bad_words = row["wrong_sen"].split()

        max_words[phenomenon] = max(
            max_words[phenomenon], max(len(good_words), len(bad_words))
        )

        all_good[phenomenon].append(good_words)
        all_bad[phenomenon].append(bad_words)

    for phenomenon in phenomena:
        if not all_good[phenomenon]:
            files[phenomenon].close()
            os.remove(f"processed/multiblimp/{lang}/{phenomenon}.csv")
        else:
            header = ["stim" + str(i + 1) for i in range(max_words[phenomenon] + 2)]
            writers[phenomenon].writerow(header)
            for i, (good_words, bad_words) in enumerate(
                zip(all_good[phenomenon], all_bad[phenomenon])
            ):
                writers[phenomenon].writerow([2 * i + 1] + good_words + ["+"])
                writers[phenomenon].writerow([2 * i + 2] + bad_words + ["-"])
            files[phenomenon].close()
