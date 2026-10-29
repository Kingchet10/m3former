"""Local rational features. Clamp is part of the differentiated function."""
import math
from functools import lru_cache

import torch

MODELS = ("cf_d1", "cf_d3", "cf_d3_schedule", "safe_d1", "safe_d3", "safe_d3_schedule")
CF_KINDS = ("local_cf", "local_safe_cf")


def protected_denominator(x, epsilon=0.1):
    # Positive sign at exactly zero; the signed function is discontinuous there.
    return torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x)) * x.abs().clamp_min(epsilon)


def continuant(values):
    """Return numerator/denominator polynomials without dividing internally."""
    previous = torch.zeros_like(values[-1])
    current = torch.ones_like(previous)
    for value in reversed(values):
        previous, current = current, value * current + previous
    return previous, current


def features(u, levels, safe, epsilon=0.1):
    values = []
    for level in levels:
        # Explicit tiny reductions permit pointwise compiler fusion, no tiny GEMM.
        a = sum(u[..., i:i+1] * level[:, i] for i in range(4)) + level[:, 4]
        values.append(1 + a.square() if safe else a)
    numerator, denominator = continuant(values)
    return numerator / (denominator if safe else protected_denominator(denominator, epsilon))


def apply(u, levels, readout, safe, epsilon=0.1, gamma=0.1):
    z = features(u, levels, safe, epsilon)
    z0 = features(torch.zeros(4, dtype=u.dtype, device=u.device), levels, safe, epsilon)
    delta = z - z0
    correction = sum(delta[..., i:i+1] * readout[:, i] for i in range(2))
    pooled = torch.stack((u[..., 0]+u[..., 1], u[..., 2]+u[..., 3]), -1) / math.sqrt(2)
    return pooled + gamma * correction


@lru_cache(None)
def compiler(dynamic=False):
    return torch.compile(apply, fullgraph=True, dynamic=dynamic)


def active_depth(completed_steps, total_steps, depth, scheduled):
    if not 0 <= completed_steps <= total_steps or total_steps <= 0:
        raise ValueError("Invalid schedule step")
    if not scheduled:
        return depth
    return sum(completed_steps >= math.ceil(total_steps * (1 - 2.0**(-i)))
               for i in range(1, depth+1))


def forward_operations(depth, safe):
    """Nominal per-group arithmetic incl. projections, pool, readout and reciprocal.

    Shared z(0) overhead is counted separately. Comparisons count as one,
    backward is estimated at twice forward by the caller, not an exact VJP count.
    """
    return 2 * depth * (8 + (2 if safe else 0)) + 4 * depth + 2 + (0 if safe else 8) + 16


def prepare_adam_state(optimizer):
    """Give DCP a stable schema even before frozen levels first receive grads.

    No dummy step: step=0/moments=0 is exactly AdamW's ordinary lazy initial
    state. This also prevents resume from expecting nonexistent frozen leaves.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("CF schedule currently validates AdamW only")
    for group in optimizer.param_groups:
        for p in group['params']:
            if not optimizer.state[p]:
                state = optimizer.state[p]
                device = p.device if group.get('capturable') or group.get('fused') else 'cpu'
                state['step'] = torch.zeros((), dtype=torch.float32, device=device)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                if group.get('amsgrad'):
                    state['max_exp_avg_sq'] = torch.zeros_like(p)
