import os
from typing import List, Dict, OrderedDict

import torch
import numpy as np
import pandas as pd
import transformers

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind, false_discovery_control
from local_datasets.local_datasets import LangLocDataset


# To cache the language mask
CACHE_DIR = os.environ.get("LOC_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))


def _get_layer(root_module, layer_name: str):
    current = root_module
    for part in layer_name.split('.'):
        if part.isdigit():
            current = current[int(part)]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            raise ValueError(f"No submodule {part!r} under {current!r}")
    return current


def _register_hook(layer: torch.nn.Module,
                    key: str,
                    target_dict: dict):
    # instantiate parameters to function defaults; otherwise they would change on next function call
    def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
        # fix for when taking out only the hidden state, this is different from dropout because of residual state
        # see:  https://github.com/huggingface/transformers/blob/c06d55564740ebdaaf866ffbbbabf8843b34df4b/src/transformers/models/gpt2/modeling_gpt2.py#L428
        output = output[0] if isinstance(output, (tuple, list)) else output
        target_dict[key] = output

    hook = layer.register_forward_hook(hook_function)
    return hook


def setup_hooks(model, layer_names):
    hooks = []
    layer_representations = OrderedDict()

    for layer_name in layer_names:
        layer = _get_layer(model, layer_name)
        hook = _register_hook(layer, key=layer_name,
                              target_dict=layer_representations)
        hooks.append(hook)

    return hooks, layer_representations


def extract_batch(
    model: torch.nn.Module, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor,
    keys: List[str],
    pooling: str = "last-token",
):
    batch_activations = {key: [] for key in keys}
    hooks, layer_representations = setup_hooks(model, keys)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    for sample_idx in range(len(input_ids)):
        for key in keys:
            if pooling == "mean":
                activations = layer_representations[key][sample_idx].mean(dim=0).cpu()
            elif pooling == "sum":
                activations = layer_representations[key][sample_idx].sum(dim=0).cpu()
            else:
                activations = layer_representations[key][sample_idx][-1].cpu()    
            batch_activations[key].append(activations)

    for hook in hooks:
        hook.remove()

    return batch_activations


def extract_representations(
    dirpath: str,
    network: str,
    pooling: str,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    keys: List[int],
    hidden_dim: int,
    batch_size: int,
    device: torch.device,
    start: int,
    end: int,
) -> Dict[str, Dict[str, np.array]]:

    loc_dataset = LangLocDataset(dirpath, network, start, end)
    langloc_dataloader = DataLoader(loc_dataset, batch_size=batch_size, num_workers=0)

    print(f"> Using Device: {device}")
    model.eval()
    model.to(device)

    # Allocate storage using the same keys.
    final_layer_representations = {
        "positive": {key: np.zeros((len(loc_dataset.positive), hidden_dim)) for key in keys},
        "negative": {key: np.zeros((len(loc_dataset.negative), hidden_dim)) for key in keys}
    }
    
    for batch_idx, batch_data in tqdm(enumerate(langloc_dataloader), total=len(langloc_dataloader)):
        sents, non_words = batch_data
        sent_tokens = tokenizer(sents, return_tensors='pt').to(device)
        non_words_tokens = tokenizer(non_words, return_tensors='pt').to(device)
        
        # **Pass the computed 'keys' list rather than the original layer_names.**
        batch_real_actv = extract_batch(model, sent_tokens["input_ids"], sent_tokens["attention_mask"], keys, pooling)
        batch_rand_actv = extract_batch(model, non_words_tokens["input_ids"], non_words_tokens["attention_mask"], keys, pooling)

        for key in keys:
            final_layer_representations["positive"][key][batch_idx*batch_size:(batch_idx+1)*batch_size] = \
                torch.stack(batch_real_actv[key]).numpy()
            final_layer_representations["negative"][key][batch_idx*batch_size:(batch_idx+1)*batch_size] = \
                torch.stack(batch_rand_actv[key]).numpy()

    return final_layer_representations


def localize(model_id: str,
    dirpath: str,
    network: str,
    pooling: str,
    model: torch.nn.Module,
    num_units: int, 
    tokenizer: transformers.PreTrainedTokenizer, 
    hidden_dim: int, 
    layer_names: List[str], 
    batch_size: int,
    device: torch.device,
    start: int = 0,
    end: int = None,
    percentage: float = None,
    overwrite: bool = False,
    submodules: bool = False,
):
    if not end:
        df = pd.read_csv(f"{dirpath}/{network}.csv")
        end = len(df)

    network_for_saving = network.replace("/", "_")
    
    save_path = f"{CACHE_DIR}/{model_id}_network={network_for_saving}_perc={percentage}_data_range={start}:{end}.npy"
    save_path_pvalues = f"{CACHE_DIR}/{model_id}_network={network_for_saving}_perc={percentage}_data_range={start}:{end}_pvalues.npy"

    if os.path.exists(save_path) and not overwrite:
        print(f"> Loading mask from {save_path}")
        return np.load(save_path)

    num_layers = len(layer_names)
    keys = layer_names

    representations = extract_representations(
        dirpath=dirpath,
        network=network, 
        pooling=pooling,
        model=model, 
        tokenizer=tokenizer, 
        keys=keys,
        hidden_dim=hidden_dim, 
        batch_size=batch_size,
        device=device,
        start=start,
        end=end,
    )

    # Use the same 'keys' to allocate and iterate over the activations:
    p_values_matrix = np.zeros((len(keys), hidden_dim))
    t_values_matrix = np.zeros((len(keys), hidden_dim))

    for layer_idx, key in tqdm(enumerate(keys), total=len(keys)):
        positive_actv = np.abs(representations["positive"][key])
        negative_actv = np.abs(representations["negative"][key])

        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = ttest_ind(positive_actv, negative_actv, axis=0, equal_var=False)
    
    def is_topk(a, k=1):
        _, rix = np.unique(-a, return_inverse=True)
        if submodules:
            return (np.where(rix < k, 1, 0).reshape(a.shape), rix.reshape(a.shape))
        return np.where(rix < k, 1, 0).reshape(a.shape)

    num_units = int((percentage/100) * hidden_dim * len(keys))
    print(f"> Percentage: {percentage}% --> Num Units: {num_units}")

    output = is_topk(t_values_matrix, k=num_units)
    language_mask = output[0] if submodules else output

    print(f"> Num units: {language_mask.sum()}")
    p_values_flat = p_values_matrix.flatten()
    num_layers, num_units = p_values_matrix.shape
    # Clip values to [0,1] range
    p_values_flat = np.clip(p_values_flat, 0, 1)
    # Replace any nan/inf with 1 (most conservative p-value)
    p_values_flat = np.nan_to_num(p_values_flat, nan=1.0, posinf=1.0, neginf=1.0)
    adjusted_p_values = false_discovery_control(p_values_flat)
    adjusted_p_values = adjusted_p_values.reshape((num_layers, num_units))
    np.save(save_path, language_mask)
    np.save(save_path_pvalues, adjusted_p_values)
    print(f"> {model_id} {network} mask cached to {save_path}")

    return output

