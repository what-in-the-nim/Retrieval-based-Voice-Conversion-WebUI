import pickle
import time
from collections import OrderedDict
from io import BytesIO

import torch
from tqdm import tqdm


def load_inputs(path: str, device: str = "cpu", is_half: bool = False) -> dict:
    """Load inputs from given path and cast to device."""
    # Load parmeters into CPU first.
    params = torch.load(path, map_location="cpu")
    # Iterate through parameters
    for key in params:
        # Cast parameter to device
        params[key] = params[key].to(device)
        # If half precision is enabled and parameter is float32, cast to float16
        if is_half and params[key].dtype == torch.float32:
            params[key] = params[key].half()
        # If half precision is disabled and parameter is float16, cast to float32
        elif not is_half and params[key].dtype == torch.float16:
            params[key] = params[key].float()
    return params


def benchmark(
    model,
    inputs_path: str,
    device: str = "cpu",
    epoch: int = 1000,
    is_half: bool = False,
) -> None:
    """Benchmark model inference time."""
    # Load model inputs
    inputs = load_inputs(inputs_path, device, is_half)
    # Set the time counter to zero.
    total_ts = 0.0
    for _ in tqdm(range(epoch)):
        # Measure inference time.
        start_time = time.perf_counter()
        model(**inputs)
        stop_time = time.perf_counter()
        # Accumulate inference time.
        total_ts += stop_time - start_time

    print(f"num_epoch: {epoch} | avg time(ms): {(total_ts*1000)/epoch}")


def export(
    model: torch.nn.Module,
    mode: str = "trace",
    inputs: dict = None,
    device=torch.device("cpu"),
    is_half: bool = False,
) -> dict:
    model = model.half() if is_half else model.float()
    model.eval()
    if mode == "trace":
        assert inputs is not None
        model_jit = torch.jit.trace(model, example_kwarg_inputs=inputs)
    elif mode == "script":
        model_jit = torch.jit.script(model)
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    buffer = BytesIO()
    # model_jit=model_jit.cpu()
    torch.jit.save(model_jit, buffer)
    del model_jit
    cpt = OrderedDict()
    cpt["model"] = buffer.getvalue()
    cpt["is_half"] = is_half
    return cpt


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save(ckpt: dict, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(ckpt, f)


def rmvpe_jit_export(
    model_path: str,
    mode: str = "script",
    inputs_path: str = None,
    save_path: str = None,
    device=torch.device("cpu"),
    is_half=False,
):
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")
    from .get_rmvpe import get_rmvpe

    model = get_rmvpe(model_path, device)
    inputs = None
    if mode == "trace":
        inputs = load_inputs(inputs_path, device, is_half)
    ckpt = export(model, mode, inputs, device, is_half)
    ckpt["device"] = str(device)
    save(ckpt, save_path)
    return ckpt


def synthesizer_jit_export(
    model_path: str,
    mode: str = "script",
    inputs_path: str = None,
    save_path: str = None,
    device=torch.device("cpu"),
    is_half=False,
):
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")
    from .get_synthesizer import get_synthesizer

    model, cpt = get_synthesizer(model_path, device)
    assert isinstance(cpt, dict)
    model.forward = model.infer
    inputs = None
    if mode == "trace":
        inputs = load_inputs(inputs_path, device, is_half)
    ckpt = export(model, mode, inputs, device, is_half)
    cpt.pop("weight")
    cpt["model"] = ckpt["model"]
    cpt["device"] = device
    save(cpt, save_path)
    return cpt
