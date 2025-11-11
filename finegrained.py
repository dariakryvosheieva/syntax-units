import os
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from localize import localize
from model_utils import get_layer_names, get_hidden_dim


CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--savedir", type=str, required=True)
    argparser.add_argument("--percentage", type=float, default=1)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument(
        "--pooling", type=str, default="last-token", choices=["last-token", "mean"]
    )

    args = argparser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    savedir = args.savedir
    percentage = args.percentage
    pooling = args.pooling
    np.random.seed(args.seed)

    print(f"> Running with model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_name = os.path.basename(model_name)
    num_blocks = len(get_layer_names(model_name, parts="block"))
    layer_names = get_layer_names(model_name, parts="submodules")

    results = {}
    dirpath = f"benchmarks/processed/{dataset}"

    for suite in os.listdir(dirpath):
        network = suite[:-4]
        results[network] = []

        language_network, rix = localize(
            model_id=model_name,
            dirpath=dirpath,
            network=network,
            pooling=pooling,
            model=model,
            num_units=None,
            tokenizer=tokenizer,
            hidden_dim=get_hidden_dim(model_name),
            layer_names=layer_names,
            batch_size=1,
            device=device,
            percentage=percentage,
            submodules=True,
        )

        for i in range(rix.shape[0]):
            is_attn = i < num_blocks
            layer_idx = i if is_attn else (i - num_blocks)
            kind = "attn" if is_attn else "mlp"

            for j in range(rix.shape[1]):
                if language_network[i][j]:
                    results[network].append((int(rix[i][j]), layer_idx, kind, j))

    with open(
        f"{savedir}/finegrained_{dataset}_{model_name}_{percentage}%.txt", "w"
    ) as f:
        to_write = ""
        for network in results.keys():
            to_write += f"{network}\n"
            results[network].sort()
            for element in results[network]:
                to_write += f"{element[0]} {element[1]} {element[2]} {element[3]}\n"
        f.write(to_write)
