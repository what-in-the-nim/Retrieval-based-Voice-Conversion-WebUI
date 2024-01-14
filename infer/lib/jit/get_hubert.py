import os

import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.hubert import HubertModel

def get_hubert_model(
    path: str = "assets/hubert/hubert_base.pt",
    device: str | torch.device = torch.device("cpu"),
) -> HubertModel:
    """Load Huber model from given path and cast to device."""
    # Check if model exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hubert model not found at {path}")
    # Load Fairseq Hubert model
    models, _, _ = load_model_ensemble_and_task([path])
    model: HubertModel = models[0]
    # Cast model to device
    model = model.to(device)

    # Add `infer` method to model
    def infer(source, padding_mask, output_layer: torch.Tensor):
        output_layer = output_layer.item()
        logits = model.extract_features(
            source=source, padding_mask=padding_mask, output_layer=output_layer
        )
        feats = model.final_proj(logits[0]) if output_layer == 9 else logits[0]
        return feats

    model.infer = infer
    return model
