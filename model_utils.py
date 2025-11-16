def get_num_blocks(model_name):
    return {
        "gpt2-xl": 48,
        "Llama-3.2-3B": 28,
        "gemma-3-4b-pt": 34,
        "deepseek-llm-7b-base": 30,
        "Phi-4-mini-instruct": 32,
        "Falcon3-3B-Base": 22,
        "Mistral-7B-v0.3": 32,
    }[model_name]


def get_hidden_dim(model_name):
    return {
        "gpt2-xl": 1600,
        "Llama-3.2-3B": 3072,
        "gemma-3-4b-pt": 2560,
        "deepseek-llm-7b-base": 4096,
        "Phi-4-mini-instruct": 3072,
        "Falcon3-3B-Base": 3072,
        "Mistral-7B-v0.3": 4096,
    }[model_name]


def get_layer_names(model_name, *, parts="block"):
    num_blocks = get_num_blocks(model_name)

    # GPT-2: transformer.h.<i>.{attn,mlp}
    if "gpt2" in model_name:
        if parts == "block":
            return [f"transformer.h.{i}" for i in range(num_blocks)]
        elif parts == "submodules":
            return [f"transformer.h.{i}.attn" for i in range(num_blocks)] + [
                f"transformer.h.{i}.mlp" for i in range(num_blocks)
            ]
    # LLaMA / Falcon / DeepSeek / Phi / Mistral: model.layers.<i>.{self_attn,mlp}
    elif any(
        x in model_name for x in ["Llama", "Falcon", "deepseek", "Phi", "Mistral", "gemma"]
    ):
        if parts == "block":
            return [f"model.layers.{i}" for i in range(num_blocks)]
        elif parts == "submodules":
            return [f"model.layers.{i}.self_attn" for i in range(num_blocks)] + [
                f"model.layers.{i}.mlp" for i in range(num_blocks)
            ]
    # Gemma: model.language_model.layers.<i>.{self_attn,mlp}
    elif "gemma" in model_name:
        if parts == "block":
            return [f"model.language_model.layers.{i}" for i in range(num_blocks)]
        elif parts == "submodules":
            return [
                f"model.language_model.layers.{i}.self_attn" for i in range(num_blocks)
            ] + [f"model.language_model.layers.{i}.mlp" for i in range(num_blocks)]
    else:
        raise ValueError(f"{model_name} not supported currently!")
