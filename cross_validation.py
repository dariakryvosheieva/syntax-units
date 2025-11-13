import os
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_utils import get_num_blocks, get_hidden_dim
from utils import cross_validation


CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--savedir", type=str, required=True)
    argparser.add_argument("--percentage", type=float, default=1.0)
    argparser.add_argument("--num-folds", type=int, default=2)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument(
        "--pooling", type=str, default="last-token", choices=["last-token", "mean"]
    )
    args = argparser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    savedir = args.savedir
    percentage = args.percentage
    num_folds = args.num_folds
    pooling = args.pooling
    np.random.seed(args.seed)

    print(f"> Running with model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)
    model.eval()

    model_name = os.path.basename(model_name)

    results = []
    for suite in os.listdir(f"benchmarks/processed/{dataset}"):
        language_overlap = np.sum(
            cross_validation(
                model_name,
                model,
                tokenizer,
                dataset,
                suite[:-4],
                num_folds,
                pooling,
                device,
                percentage,
            )
        )
        results.append((language_overlap, suite[:-4]))
    results.sort(reverse=True)

    num_units = get_num_blocks(model_name) * get_hidden_dim(model_name)
    percentage_units = num_units * args.percentage // 100
    print(np.mean([line[0] / percentage_units * 100 for line in results]))
    with open(
        f"{savedir}/cross-validation_{dataset}_{model_name}_{percentage}%_{num_folds}-fold.txt",
        "w",
    ) as f:
        to_write = ""
        for line in results:
            to_write += (
                f"{line[0]} ({line[0] / percentage_units * 100:.2f}%) {line[1]}\n"
            )
        f.write(to_write)
