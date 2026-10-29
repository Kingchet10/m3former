"""DPR pairing only; frozen Safe-rational is imported unchanged."""
import functools
import torch
import torch.nn.functional as F
from .local_functions import apply as rational_apply


def fractional_shift(b, delta, *, hard=False):
    """Positive delta reads i+delta: integer case torch.roll(b,-k).

    Last axis is cyclic; delta has one value per preceding index. Integer
    knots use the right derivative, including zero initialization. No STE.
    """
    n = b.shape[-1]
    if hard:
        delta = torch.round(delta)  # ties-to-even
    k = torch.floor(delta)
    alpha = (delta-k).unsqueeze(-1)
    indices = (torch.arange(n, device=b.device)+k.long().unsqueeze(-1)) % n
    left = torch.gather(b,-1,indices)
    if hard:
        return left
    right = torch.gather(b,-1,(indices+1) % n)
    return (1-alpha)*left+alpha*right


def shifts(x, weight, bias, block_width):
    scale = block_width/2
    return scale*torch.tanh(F.linear(x,weight,bias)/scale)


def compress(p, delta, theta, *, hard=False):
    groups = delta.shape[-1]
    if groups not in (1,4) or p.shape[-1] % (2*groups):
        raise ValueError('DPR requires one or four equal even/odd stream blocks')
    a = p[...,0::2].unflatten(-1,(groups,-1))
    b = p[...,1::2].unflatten(-1,(groups,-1))
    paired = torch.stack((a,fractional_shift(b,delta,hard=hard)),dim=-1)
    return rational_apply(paired,theta,'safe_rational').flatten(-2)


@functools.lru_cache(None)
def compiler(dynamic=False):
    return torch.compile(compress,fullgraph=True,dynamic=dynamic)
