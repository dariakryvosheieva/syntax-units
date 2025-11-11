from glob import glob

import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def process(sent, orig_word=None, new_word=None):
    # split into words
    sent = sent.split()

    # trim <eos>
    sent = sent[:-1]

    # word replacement (if needed)
    if orig_word is not None:
        index = sent.index(orig_word)
        sent[index] = new_word

    # attach punctuation to previous word
    punct = ".,!?:;)"
    i = 0
    while i < len(sent):
        if i > 0 and sent[i] in punct:
            sent[i - 1] += sent[i]
            sent.pop(i)
            i -= 1
        elif i < len(sent) - 1 and sent[i] == "(":
            sent[i + 1] = sent[i] + sent[i + 1]
            sent.pop(i)
            i -= 1
        i += 1

    # "n't" and "'s" handling
    i = 0
    while i < len(sent):
        if i > 0 and (sent[i] == "n't" and sent[i - 1] == "do" or sent[i] == "'s"):
            sent[i - 1] += sent[i]
            sent.pop(i)
            i -= 1
        else:
            i += 1

    return sent


class LangLocDataset(Dataset):
    def __init__(self, dirpath, network, start, end):
        self.network = network

        if self.network == "langloc":
            paths = glob(f"{dirpath}/*.csv")
            data = pd.read_csv(paths[0])
            for path in paths[1:]:
                run_data = pd.read_csv(path)
                data = pd.concat([data, run_data])

            data = data.iloc[start:end].reset_index(drop=True)

            data["sent"] = data["stim2"].apply(str.lower)

            for stimuli_idx in range(3, 14):
                data["sent"] += " " + data[f"stim{stimuli_idx}"].apply(str.lower)

            self.positive = data[data["stim14"] == "S"]["sent"]
            self.negative = data[data["stim14"] == "N"]["sent"]
        else:
            path = f"{dirpath}/{network}.csv"

            whole_data = pd.read_csv(path, dtype=str)

            if self.network in [
                "semantic",
                "agreement",
                "tense",
                "spelling",
                "torontofantasy",
            ]:
                # Get number of pairs
                n_pairs = whole_data.shape[0] // 2

                # Generate shuffled pair indices
                indices = np.random.permutation(n_pairs)

                # Create new index order that keeps pairs together
                new_order = np.array([[2 * i, 2 * i + 1] for i in indices]).flatten()

                # Reindex the dataframe
                whole_data = whole_data.iloc[new_order].reset_index(drop=True)

            data = whole_data.iloc[start:end].reset_index(drop=True)
            data = data.fillna("")

            max_len = data.shape[1]

            data["sent"] = data["stim2"]

            for stimuli_idx in range(3, max_len + 1):
                data["sent"] += " " + data[f"stim{stimuli_idx}"]

            for i in range(data.shape[0]):
                data.at[i, "sent"] = data.at[i, "sent"].strip()[:-2]
                if self.network == "sentences_wordlists":
                    data.at[i, "sent"] = data.at[i, "sent"].lower()

            self.positive = []
            self.negative = []

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if data.iloc[i, j] == "+":
                        self.positive.append(data.iloc[i]["sent"])
                        break
                    if data.iloc[i, j] == "-":
                        self.negative.append(data.iloc[i]["sent"])
                        break

    def __getitem__(self, idx):
        if self.network == "langloc":
            return self.positive.iloc[idx].strip(), self.negative.iloc[idx].strip()
        else:
            return self.positive[idx].strip(), self.negative[idx].strip()

    def __len__(self):
        return min(len(self.positive), len(self.negative))
