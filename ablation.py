import os
import argparse

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from localize import localize, extract_batch
from benchmarks.benchmarks import LangLocDataset
from model_utils import get_layer_names, get_hidden_dim
from utils import random_mask_from_remaining, evaluate

from models.modeling_gpt2 import GPT2LMHeadModel
from models.modeling_llama import LlamaForCausalLM
from models.modeling_phi3 import Phi3ForCausalLM
from models.modeling_gemma3 import Gemma3ForCausalLM
from models.modeling_mistral import MistralForCausalLM


CACHE_DIR = os.environ.get("LOC_CACHE", f"cache")

device = "cuda" if torch.cuda.is_available() else "cpu"


def global_activation_mean(
        model,
        tokenizer,
        dataset: str,
        network: str,
        layer_keys: list[str],
        device,
        batch_size: int,
        start_idx: int,
        end_idx: int,
    ) -> float:
        from torch.utils.data import DataLoader
        import torch

        dirpath = f"benchmarks/processed/{dataset}"
        df = pd.read_csv(f"{dirpath}/{network}.csv")
        start_idx = max(0, start_idx)
        end_idx = min(len(df), end_idx)
        if start_idx >= end_idx:
            return 0.0

        loc_dataset = LangLocDataset(dirpath, network, start_idx, end_idx)
        loader = DataLoader(loc_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

        running_sum = 0.0
        running_count = 0

        model.eval()
        model.to(device)
        with torch.no_grad():
            for (sents, _non_words) in loader:
                enc = tokenizer(list(sents), return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

                batch_act = extract_batch(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    keys=layer_keys,
                    pooling="last-token",
                )
                for key in layer_keys:
                    v = torch.stack(batch_act[key])  # [B, H]
                    running_sum += v.sum().item()
                    running_count += v.numel()

        return float(running_sum / running_count) if running_count else 0.0


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ablation-type", type=str, default="zero", choices=["zero", "mean"])
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--savedir", type=str, required=True)
    argparser.add_argument("--percentage", type=float, default=1.0)
    argparser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789",
    )
    argparser.add_argument(
        "--pooling", type=str, default="last-token", choices=["last-token", "mean"]
    )

    args = argparser.parse_args()

    ablation_type = args.ablation_type
    model_name = args.model_name
    dataset = args.dataset
    savedir = args.savedir
    percentage = args.percentage
    pooling = args.pooling

    random_seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    n_random = len(random_seeds)

    print(f"> Running with model {model_name}")

    if "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif "Llama" in model_name or "deepseek" in model_name or "Falcon" in model_name:
        model = LlamaForCausalLM.from_pretrained(model_name)
    elif "Phi" in model_name:
        model = Phi3ForCausalLM.from_pretrained(model_name)
    elif "gemma" in model_name:
        model = Gemma3ForCausalLM.from_pretrained(model_name)
    elif "Mistral" in model_name:
        model = MistralForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_name = os.path.basename(model_name)

    accs_none, accs_language, accs_random = {}, {}, {}

    for suite in os.listdir(f"benchmarks/processed/{dataset}"):
        network = suite[:-4]

        model.set_language_selective_mask(None)
        acc_none = evaluate(
            model, tokenizer, dataset, network, device, second_half=True
        )

        length = len(pd.read_csv(f"benchmarks/processed/{dataset}/{network}.csv"))
        
        if ablation_type == "mean":
             global_mu = global_activation_mean(
                 model=model,
                 tokenizer=tokenizer,
                 dataset=dataset,
                 network=network,
                 layer_keys=get_layer_names(model_name),
                 batch_size=1,
                 device=device,
                 start_idx=0,
                 end_idx=(length // 2),
             )
             print(f"Global mean: {global_mu}")

        language_mask = localize(
            model_id=model_name,
            dirpath=f"benchmarks/processed/{dataset}",
            network=network,
            pooling=pooling,
            model=model,
            num_units=None,
            tokenizer=tokenizer,
            hidden_dim=get_hidden_dim(model_name),
            layer_names=get_layer_names(model_name),
            batch_size=1,
            device=device,
            start=0,
            end=(length // 2),
            percentage=percentage,
        )
        model.set_language_selective_mask(language_mask)
        
        if ablation_type == "mean":
            model.set_ablation_mode(mode="mean", fill_value=global_mu)
        else:
            model.set_ablation_mode(mode="zero", fill_value=0.0)
         
        acc_language = evaluate(
            model, tokenizer, dataset, network, device, second_half=True
        )

        random_scores = []
        for s in random_seeds:
            np.random.seed(s)
            try:
                import torch

                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)
            except Exception:
                pass

            model.set_language_selective_mask(random_mask_from_remaining(language_mask))
            
            if ablation_type == "mean":
                model.set_ablation_mode(mode="mean", fill_value=global_mu)
            else:
                model.set_ablation_mode(mode="zero", fill_value=0.0)
            
            acc_random = evaluate(
                model, tokenizer, dataset, network, device, second_half=True
            )
            random_scores.append(acc_random)

        accs_none[network] = acc_none
        accs_language[network] = acc_language
        accs_random[network] = float(np.mean(random_scores))

    with open(f"{savedir}/ablation_{dataset}_{model_name}_{percentage}%.txt", "w") as f:
        to_write = ""
        for suite in os.listdir(f"benchmarks/processed/{dataset}"):
            network = suite[:-4]
            to_write += f"{network}: {accs_none[network]:.3f}/{accs_language[network]:.3f}/{accs_random[network]:.3f}\n"
        to_write += f"Average performance (original model): {np.mean(list(accs_none.values()))}\n"
        to_write += f"Average performance (top {percentage}% ablated): {np.mean(list(accs_language.values()))}\n"
        to_write += f"Average performance (random {percentage}% ablated): {np.mean(list(accs_random.values()))}\n"
        f.write(to_write)
