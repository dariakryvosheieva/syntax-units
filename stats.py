import os

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from localize import localize
from model_utils import get_num_blocks, get_hidden_dim, get_layer_names


MODEL_NAMES = [
    "gpt2-xl",
    "Llama-3.2-3B",
    "Falcon3-3B-Base",
    "Phi-4-mini-instruct",
    "gemma-3-4b-pt",
    "deepseek-llm-7b-base",
    "Mistral-7B-v0.3"
]


def synt_lex_overlap():
    means = []

    for model_name in MODEL_NAMES:
        one_pct_units = int(get_num_blocks(model_name) * get_hidden_dim(model_name) * 0.01)

        n_path = f"english/cross-overlap/cross-overlap_blimp_blimp-n-sub-concat_{model_name}_1.0%.csv"
        v_path = f"english/cross-overlap/cross-overlap_blimp_blimp-v-sub-concat_{model_name}_1.0%.csv"
        
        if not (os.path.exists(n_path) and os.path.exists(v_path)):
            continue
        
        n_df = pd.read_csv(n_path, index_col=0)
        v_df = pd.read_csv(v_path, index_col=0)
        
        n_df = n_df[n_df.index.astype(str).str.contains("agreement", case=False, na=False)]
        v_df = v_df[v_df.index.astype(str).str.contains("agreement", case=False, na=False)]
        
        n_df["blimp_n_sub"] /= one_pct_units
        v_df["blimp_v_sub"] /= one_pct_units
        
        print(n_df)

        avg_df = pd.DataFrame({
            "avg_blimp_sub": (
                n_df["blimp_n_sub"].reset_index(drop=True)
                + v_df["blimp_v_sub"].reset_index(drop=True)
            ) / 2
        })

        means.append(avg_df["avg_blimp_sub"].mean())

    return np.mean(means)


def cross_overlap_grand(dataset, percentage=1.0, seed=42, pooling="last-token"):
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    networks = [fn[:-4] for fn in os.listdir(f"benchmarks/processed/{dataset}") if fn.endswith(".csv")]
    networks = sorted(networks)

    results = {}

    for model_name in MODEL_NAMES:
        print(f"> Running with model {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        base_name = os.path.basename(model_name)
        model.to(device)
        model.eval()

        hidden_dim = get_hidden_dim(base_name)
        layer_names = get_layer_names(base_name)

        first_mask = localize(
            base_name,
            f"benchmarks/processed/{dataset}",
            networks[0],
            pooling,
            model,
            num_units=None,
            tokenizer=tokenizer,
            hidden_dim=hidden_dim,
            layer_names=layer_names,
            batch_size=1,
            device=device,
            percentage=percentage,
        )
        unit_counts = np.zeros_like(first_mask, dtype=np.int32)

        for net in networks:
            mask = localize(
                base_name,
                f"benchmarks/processed/{dataset}",
                net,
                pooling,
                model,
                num_units=None,
                tokenizer=tokenizer,
                hidden_dim=hidden_dim,
                layer_names=layer_names,
                batch_size=1,
                device=device,
                percentage=percentage,
            )
            unit_counts += mask.astype(np.int32)

        X = int(unit_counts.max())
        print(f"{model_name}: The most syntax-general unit is shared among {X} phenomena.")
        results[model_name] = X

    return results


def zero_mean_correlation():
    correlations = []

    for model_name in MODEL_NAMES:
        zero_ablation_scores = {}
        with open(f"english/ablation/ablation_blimp_{model_name}_1.0%.txt", "r") as f:
            lines = [line.split() for line in f.readlines()]
            for line in lines:
                if len(line) == 2:
                    suite = line[0]
                    original, top, random = map(float, line[1].split("/"))
                    zero_ablation_scores[suite[:-1]] = top - original
        mean_ablation_scores = {}
        with open(f"english/mean-ablation/ablation_blimp_{model_name}_1.0%.txt", "r") as f:
            lines = [line.split() for line in f.readlines()]
            for line in lines:
                if len(line) == 2:
                    suite = line[0]
                    original, top, random = map(float, line[1].split("/"))
                    mean_ablation_scores[suite[:-1]] = top - original
        
        sorted_zero_ablation_scores = [x[1] for x in sorted(zero_ablation_scores.items())]
        sorted_mean_ablation_scores = [x[1] for x in sorted(mean_ablation_scores.items())]

        correlation = np.corrcoef(sorted_zero_ablation_scores, sorted_mean_ablation_scores)[0, 1]
        print(f"{model_name}: {correlation}")
        correlations.append(correlation)

    return np.mean(correlations)


if __name__ == "__main__":
    print(zero_mean_correlation())