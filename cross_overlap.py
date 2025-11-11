import os
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import cross_overlap, new_matrix


CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--dataset-1", type=str, required=True)
    argparser.add_argument("--dataset-2", type=str, required=True)
    argparser.add_argument("--savedir", type=str, required=True)
    argparser.add_argument("--percentage", type=float, default=1)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument(
        "--pooling", type=str, default="last-token", choices=["last-token", "mean"]
    )

    args = argparser.parse_args()

    model_name = args.model_name
    dataset_1 = args.dataset_1
    dataset_2 = args.dataset_2
    savedir = args.savedir
    percentage = args.percentage
    pooling = args.pooling
    np.random.seed(args.seed)

    print(f"> Running with model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_name = os.path.basename(model_name)

    model.to(device)
    model.eval()

    networks_1 = [
        suite[:-4] for suite in os.listdir(f"local_datasets/processed/{dataset_1}")
    ]
    networks_2 = [
        suite[:-4] for suite in os.listdir(f"local_datasets/processed/{dataset_2}")
    ]

    language_matrix = new_matrix(networks_1, networks_2)

    for network_1 in networks_1:
        for network_2 in networks_2:
            language_matrix.loc[network_1, network_2] = np.sum(
                cross_overlap(
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
                )
            )

    language_matrix.to_csv(
        f"{savedir}/cross-overlap_{dataset_1}_{dataset_2}_{model_name}_{percentage}%.csv"
    )
