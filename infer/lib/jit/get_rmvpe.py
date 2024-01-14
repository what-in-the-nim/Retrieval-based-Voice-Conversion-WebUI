import torch

from infer.lib.rmvpe import E2E

def get_rmvpe(
    path: str = "assets/rmvpe/rmvpe.pt",
    device: str | torch.device = "cpu",
) -> E2E:
    """Load RMVPE model from given path and cast to device."""
    # Initialize model
    model = E2E(4, 1, (2, 2))
    # Load model state dict to model
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    # Set model to eval mode.
    model.eval()
    # Cast model to device
    model = model.to(device)
    return model
