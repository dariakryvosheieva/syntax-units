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
    argparser.add_argument(
        "--phenomenon",
        type=str,
        required=True,
        choices=["SV-P", "SV-#", "SV-G", "SP-P", "SP-#", "SP-G"],
    )
    argparser.add_argument("--percentage", type=float, default=1)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument(
        "--pooling", type=str, default="last-token", choices=["last-token", "mean"]
    )

    args = argparser.parse_args()

    model_name = args.model_name
    phenomenon = args.phenomenon
    percentage = args.percentage
    pooling = args.pooling
    np.random.seed(args.seed)

    print(f"> Running with model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_name = os.path.basename(model_name)

    model.to(device)
    model.eval()

    networks = []
    for root, dirs, files in os.walk("benchmarks/processed/multiblimp"):
        for file in files:
            if file == f"{phenomenon}.csv":
                networks.append(os.path.basename(root))

    language_matrix = new_matrix(networks, networks)

    for network_1 in networks:
        for network_2 in networks:
            language_matrix.loc[network_1, network_2] = np.sum(
                cross_overlap(
                    model_name,
                    model,
                    tokenizer,
                    "multiblimp",
                    "multiblimp",
                    f"{network_1}/{phenomenon}",
                    f"{network_2}/{phenomenon}",
                    pooling,
                    device,
                    percentage,
                )
            )

    language_matrix.to_csv(
        f"multilingual/cross-overlap_multiblimp_{phenomenon}_{model_name}_{percentage}%.csv"
    )
