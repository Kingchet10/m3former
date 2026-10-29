"""Anchor-preserving, two-output dendritic readout; ordinary PyTorch algebra."""

import torch
import torch.nn.functional as F

KINDS = ("s2_l", "s2_q", "s2_blocked", "s2_mid")
PAIRS = tuple((i, j) for i in range(4) for j in range(i, 4))
BLOCK_PAIRS = (((0, 0), (0, 1), (1, 1)), ((2, 2), (2, 3), (3, 3)))


def initial_a(*, device=None, dtype=None):
    return (
        torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], device=device, dtype=dtype) / 2**0.5
    )


def q_count(kind):
    if kind not in KINDS:
        raise ValueError("Unknown S2 readout")
    return 0 if kind == "s2_l" else (6 if kind == "s2_blocked" else 20)


def _features(p, pairs):
    return torch.stack([p[..., i] * p[..., j] for i, j in pairs], dim=-1)


def _weights(q, pairs):
    return q * q.new_tensor([1 if i == j else 2 for i, j in pairs])


def readout(p, a, q=None, *, blocked=False):
    """p[..., G, 4] -> [..., G, 2]; q stores actual upper-triangular Q entries.

    Precision follows input, including FP64 oracle/gradcheck. The calling FFN
    disables autocast and explicitly supplies FP32 inputs in mixed precision.
    Blocked Q has six trainable entries; A remains a full learnable 2x4 matrix.
    """
    if p.shape[-1] != 4 or a.shape != (2, 4):
        raise ValueError("S2 requires B=4, K=2")
    linear = F.linear(p, a)
    quadratic = torch.zeros_like(linear)
    if q is not None:
        if q.numel() != (6 if blocked else 20):
            raise ValueError("Invalid compact Q size")
        if blocked:
            quadratic = torch.cat(
                [
                    F.linear(
                        _features(p, pairs), _weights(q.reshape(2, 3)[k : k + 1], pairs)
                    )
                    for k, pairs in enumerate(BLOCK_PAIRS)
                ],
                dim=-1,
            )
        else:
            quadratic = F.linear(_features(p, PAIRS), _weights(q.reshape(2, 10), PAIRS))
    return linear + quadratic, linear, quadratic


def symmetric_q(q, *, blocked=False):
    """Expand for diagnostic/oracle use, preserving autograd."""
    result = q.new_zeros(2, 4, 4)
    for k in range(2):
        pairs = BLOCK_PAIRS[k] if blocked else PAIRS
        for index, (i, j) in enumerate(pairs):
            value = q.reshape(2, -1)[k, index]
            result[k, i, j] = value
            result[k, j, i] = value
    return result


def training_operations(groups, kind):
    """Direct scalar forward/VJP count, including shared A/Q reductions.

    Linear A costs 42G-8: 14G forward, 12G input VJP, 8(2G-1)
    weight VJP. Full Q adds 160G+4: feature products/VJP 50G,
    projection forward 38G, feature VJP 30G, weight VJP 20(2G-1),
    output additions 2G, off-diagonal coefficient scaling/VJP 24.
    Blocked Q performs two independent 3-feature projections: each adds
    30G-1. Product VJP accumulates both derivatives into the existing A
    gradient, including both contributions of squares. No masked GEMMs.
    Excludes indexing, casts, concatenation and implementation scatter costs.
    """
    q_count(kind)
    if groups < 1:
        raise ValueError("Positive group count required")
    return (
        42 * groups
        - 8
        + (
            0
            if kind == "s2_l"
            else (60 * groups - 2 if kind == "s2_blocked" else 160 * groups + 4)
        )
    )
