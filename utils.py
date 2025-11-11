import numpy as np
import pandas as pd
from minicons.scorer import IncrementalLMScorer

from localize import localize
from model_utils import get_layer_names, get_hidden_dim


def cross_validation(
        model_name,
        model,
        tokenizer,
        dataset,
        network,
        num_folds,
        pooling,
        device,
        percentage,
    ):

    length = len(pd.read_csv(f"local_datasets/processed/{dataset}/{network}.csv"))
    sents_per_fold = length // num_folds

    language_intersection = None

    for fold in range(num_folds):
        start = fold * sents_per_fold
        end = start + sents_per_fold

        language_mask = localize(
            model_id=model_name,
            dirpath=f"local_datasets/processed/{dataset}",
            network=network,
            pooling=pooling,
            model=model,
            num_units=None,
            percentage=percentage,
            tokenizer=tokenizer,
            hidden_dim=get_hidden_dim(model_name),
            layer_names=get_layer_names(model_name),
            batch_size=1,
            device=device,
            start=start,
            end=end,
        )

        if language_intersection is None:
            language_intersection = language_mask
        else:
            language_intersection &= language_mask
    
    return language_intersection


def cross_overlap(
        model_name,
        model,
        tokenizer,
        dataset_1,
        dataset_2,
        network_1,
        network_2,
        pooling,
        device,
        percentage,
    ):

    language_mask_1 = localize(
        model_id=model_name,
        dirpath=f"local_datasets/processed/{dataset_1}",
        network=network_1,
        pooling=pooling,
        model=model,
        num_units=None,
        percentage=percentage,
        tokenizer=tokenizer,
        hidden_dim=get_hidden_dim(model_name),
        layer_names=get_layer_names(model_name),
        batch_size=1,
        device=device,
    )      

    language_mask_2 = localize(
        model_id=model_name,
        dirpath=f"local_datasets/processed/{dataset_2}",
        network=network_2,
        pooling=pooling,
        model=model,
        num_units=None,
        percentage=percentage,
        tokenizer=tokenizer,
        hidden_dim=get_hidden_dim(model_name),
        layer_names=get_layer_names(model_name),
        batch_size=1,
        device=device,
    )      
        
    return language_mask_1 & language_mask_2


def evaluate(
        model,
        tokenizer,
        dataset,
        network,
        device,
        second_half=False,
    ):
    n_correct = 0
    scr = IncrementalLMScorer(model=model, device=device, tokenizer=tokenizer)
    whole_data = pd.read_csv(f"local_datasets/processed/{dataset}/{network}.csv", dtype=str)
    if second_half:
        start = len(whole_data) // 2
        data = whole_data[start : ].reset_index(drop=True)
    else:
        data = whole_data.reset_index(drop=True)
    data = data.fillna('')
    max_len = data.shape[1]
    data["sent"] = data["stim2"]
    for stimuli_idx in range(3, max_len + 1):
        data["sent"] += " " + data[f"stim{stimuli_idx}"]
    for i in range(data.shape[0]):
        data.at[i, "sent"] = data.at[i, "sent"].strip()[:-2]
    for i in range(0, data.shape[0], 2):
        sentence_good = data.iloc[i]["sent"]
        sentence_bad = data.iloc[i + 1]["sent"]
        scores = scr.sequence_score([sentence_good, sentence_bad])
        if scores[0] > scores[1]:
            n_correct += 1
    return n_correct / (len(data) / 2)


def random_mask_from_remaining(language_mask):
    num_layers, hidden_dim = language_mask.shape
    total_num_units = np.prod(language_mask.shape)
    invlang_mask_indices = np.arange(total_num_units)[(1 - language_mask).flatten().astype(bool)]
    rand_indices = np.random.choice(invlang_mask_indices, size=int(language_mask.sum()), replace=False)
    lang_mask_rand = np.full(total_num_units, 0)
    lang_mask_rand[rand_indices] = 1
    language_mask = lang_mask_rand.reshape((num_layers, hidden_dim))
    return language_mask


def new_matrix(arr_1, arr_2):
    return pd.DataFrame(
        np.zeros((len(arr_1), len(arr_2))),
        index=arr_1,
        columns=arr_2
    )


def random_overlap_expected(n_total, k, n_folds=2):
    return n_total * (k / n_total) ** n_folds
