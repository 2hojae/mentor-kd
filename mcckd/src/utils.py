import io
import json
import os
import pickle
import random
import sys
from typing import Tuple

import torch
import numpy as np
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from torch.distributed import init_process_group


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def json_dump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def json_load(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def pickle_load(f):
    with open(f, "rb") as r:
        objects = pickle.load(r)
    return objects


def pickle_dump(obj, f):
    with open(f, "wb") as f:
        pickle.dump(obj, f)
    return f


def setup_model_parallel(use_float16=True, seed=None) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    # torch.manual_seed(1)
    set_seed(1 if seed is None else seed)
    if use_float16:
        torch.set_default_tensor_type(torch.HalfTensor)
    return local_rank, world_size


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def cross_entropy(logits, labels, weights=None, keepdim=False):
    """
    Compute Cross-Entropy Loss..
    :param logits: [batch_size, ..., num_classes] tensor.
    :param labels: [batch_size, ...] tensor. LongTensor.
    Same shape with 0th - (last - 1)th of logits.
    :param weights: [batch_size, ...] tensor, where `1` donates validate and
     `0` donates invalidate. Same shape with 0th - (last - 1)th of logits.
    :param keepdim: bool, whether not to perform reduce sum of the final result.
    :return: The mean of all examples' loss.
    """
    bzs = logits.shape[0]
    logits = logits.float()
    labels = labels.to(logits.device)
    if weights is None:
        weights = torch.ones_like(labels)
    weights = weights.float().to(logits.device)
    weights = torch.reshape(weights, [bzs, -1])
    num_classes = int(logits.size()[-1])
    logits = torch.reshape(logits, shape=[bzs, -1, num_classes])
    log_probs = F.log_softmax(logits, dim=-1)
    labels = torch.reshape(labels, [bzs, -1]).long()
    labels = F.one_hot(labels, num_classes=num_classes)
    loss = - torch.sum(log_probs * labels, dim=[-1])  # [b, s]
    if not keepdim:
        nrt = torch.sum(weights * loss, dim=-1)
        dnm = torch.sum(weights, dim=-1) + 1e-8
        loss = torch.mean(nrt / dnm, dim=0)
    return loss

EPS = 1e-7

def kl_div(source, target):
    """ Compute Kullback-Leibler divergence Loss along last dim. """
    result = source * torch.log(source + EPS) - source * torch.log(target + EPS)
    return torch.sum(result, dim=-1)


def kl_div_loss(p, q, weights=None):
    """ Compute Kullback-Leibler divergence Loss """
    p_loss = kl_div(p, q)
    q_loss = kl_div(q, p)
    if weights is not None:
        weights = weights.to(p.device)
        p_loss = p_loss * weights
        q_loss = q_loss * weights
        p_loss = p_loss.sum(dim=-1) / (weights.sum(dim=-1) + 1e-12)
        q_loss = q_loss.sum(dim=-1) / (weights.sum(dim=-1) + 1e-12)
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()
    loss = (p_loss + q_loss) / 2
    return loss


def kl_div_loss_v2(p, q, weights=None):
    """ Compute Kullback-Leibler divergence Loss """
    p_loss = kl_div(p, q)
    q_loss = kl_div(q, p)
    if weights is None:
        weights = torch.ones_like(p_loss)
    weights = weights.to(p.device)
    p_loss = p_loss * weights
    q_loss = q_loss * weights
    p_loss = p_loss.sum(dim=-1) / (weights.sum(dim=-1) + 1e-12)
    q_loss = q_loss.sum(dim=-1) / (weights.sum(dim=-1) + 1e-12)
    p_loss = p_loss.sum() / (torch.sign(weights.sum(dim=-1)).sum() + 1e-12)
    q_loss = q_loss.sum() / (torch.sign(weights.sum(dim=-1)).sum() + 1e-12)
    loss = (p_loss + q_loss) / 2
    return loss


def barrier():
    """ make sure that all other processes cannot continue until reach this op. """
    torch.distributed.barrier()
