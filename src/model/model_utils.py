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
