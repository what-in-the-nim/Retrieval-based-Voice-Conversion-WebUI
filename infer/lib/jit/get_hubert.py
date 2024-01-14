import math
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.checkpoint_utils import load_model_ensemble_and_task

# from fairseq.data.data_utils import compute_mask_indices
from fairseq.utils import index_put
from fairseq.models.hubert import HubertModel


# @torch.jit.script
def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if int(tsz % multiple) == 0:
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> torch.Tensor:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = torch.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + torch.rand([1]).item()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = torch.full([num_mask], mask_length)
        elif mask_type == "uniform":
            lengths = torch.randint(mask_other, mask_length * 2 + 1, size=[num_mask])
        elif mask_type == "normal":
            lengths = torch.normal(mask_length, mask_other, size=[num_mask])
            lengths = [max(1, int(round(x))) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = torch.randint(low=s, high=e - length, size=[1]).item()
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                t = [e - s if e - s >= length + min_space else 0 for s, e in parts]
                lens = torch.asarray(t, dtype=torch.int)
                l_sum = torch.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / torch.sum(lens)
                c = torch.multinomial(probs.float(), len(parts)).item()
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = torch.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
            mask_idc = torch.asarray(
                random.sample([i for i in range(sz - min_len)], num_mask)
            )
            mask_idc = torch.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(torch.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if isinstance(mask_idc, torch.Tensor):
            mask_idc = torch.asarray(mask_idc, dtype=torch.float)
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = torch.asarray(
                random.sample([i for i in range(mask_idc)], min_len)
            )
        if mask_dropout > 0:
            num_holes = int(round(len(mask_idc) * mask_dropout))
            mask_idc = torch.asarray(
                random.sample([i for i in range(mask_idc)], len(mask_idc) - num_holes)
            )

        mask[i, mask_idc.int()] = True

    return mask


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
