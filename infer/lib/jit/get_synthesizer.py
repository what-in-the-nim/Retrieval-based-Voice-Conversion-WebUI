import torch

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

def get_synthesizer(
    path: str,
    device: str | torch.device = "cpu",
):
    # Load state_dict inside CPU first.
    state_dict = torch.load(path, map_location="cpu")
    state_dict["config"][-3] = state_dict["weight"]["emb_g.weight"].shape[0]
    # Check if state_dict contains f0
    if_f0 = state_dict.get("f0", 1)
    version = state_dict.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*state_dict["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*state_dict["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*state_dict["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*state_dict["config"])

    del net_g.enc_q

    net_g.load_state_dict(state_dict["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, state_dict
