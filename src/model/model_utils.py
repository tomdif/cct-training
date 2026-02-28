"""Model-agnostic utilities for accessing transformer internals."""


def get_model_layers(model):
    """Return the list of transformer layers, regardless of architecture.

    Supports: Llama/Qwen/Mistral, GPTNeoX/Pythia, GPT-2.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers          # Llama, Qwen, Mistral
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers       # Pythia / GPTNeoX
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h         # GPT-2
    raise ValueError(f"Unknown model architecture: {type(model).__name__}")


def detect_lora_target_modules(model):
    """Auto-detect the correct LoRA target module names for the model architecture.

    Inspects the first transformer layer's attention block to find the
    query/value projection attribute names.

    Returns:
        list[str]: Module names suitable for peft LoraConfig target_modules.
    """
    layers = get_model_layers(model)
    layer0 = layers[0]

    # Pythia / GPTNeoX: fused QKV
    if hasattr(layer0, 'attention') and hasattr(layer0.attention, 'query_key_value'):
        return ["query_key_value"]

    # Llama / Qwen / Mistral: separate projections
    attn = getattr(layer0, 'self_attn', None)
    if attn is not None:
        modules = []
        if hasattr(attn, 'q_proj'):
            modules.append("q_proj")
        if hasattr(attn, 'v_proj'):
            modules.append("v_proj")
        if modules:
            return modules

    # GPT-2: c_attn (fused QKV as Conv1D)
    if hasattr(layer0, 'attn') and hasattr(layer0.attn, 'c_attn'):
        return ["c_attn"]

    raise ValueError(
        f"Cannot detect LoRA targets for {type(model).__name__}. "
        f"Set lora_target_modules explicitly in config."
    )
