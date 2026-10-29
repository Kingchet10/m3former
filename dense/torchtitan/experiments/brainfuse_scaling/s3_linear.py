"""Pure linear local projection: shared or independent group matrices, no Q."""

import torch
import torch.nn.functional as F

# anchor, dendritic inputs, outputs/group, mode
RECIPES = {
    "s3_light": (720, 160, 2, "shared"),
    "s3_mid": (480, 448, 2, "shared"),
    "s3_25ga": (640, 256, 2, "group_adaptive"),
    "s3_midga": (480, 448, 2, "group_adaptive"),
    "s3_k3": (640, 232, 3, "shared"),
    "s3_k3ga": (640, 232, 3, "group_adaptive"),
}


def initial_a(k, *, device=None, dtype=None):
    if k not in (2, 3):
        raise ValueError("S3 supports 4→2 and 4→3")
    rows = [[2**-0.5, 2**-0.5, 0, 0], [0, 0, 2**-0.5, 2**-0.5]]
    if k == 3:
        rows.append([0.5, -0.5, 0.5, -0.5])
    return torch.tensor(rows, device=device, dtype=dtype)


def local_linear(p, a):
    """p[..., G, 4], a[K,4] or a[G,K,4] -> [..., G, K].

    Accumulation dtype follows input; the FFN explicitly disables autocast and
    supplies FP32 (FP64 for algebra tests). No group loop or quadratic path.
    """
    if p.shape[-1] != 4 or a.shape[-1] != 4 or a.shape[-2] not in (2, 3):
        raise ValueError("Invalid local dimensions")
    if a.ndim == 2:
        return F.linear(p, a)
    if a.ndim != 3 or a.shape[0] != p.shape[-2]:
        raise ValueError("Group matrix count must match input")
    return torch.einsum("...gb,gkb->...gk", p, a)


def training_operations(tokens, groups, k, mode):
    """Scalar forward + input VJP + parameter VJP, per microbatch.

    B=4: 7K forward and 4(2K-1) input VJP per token-group.
    Parameter VJP reduces over T*G for shared A, only T for group A.
    Counts algebra, excluding reshapes/casts/monitoring/optimizer.
    """
    if min(tokens, groups) < 1 or k not in (2, 3):
        raise ValueError("Invalid operation dimensions")
    common = tokens * groups * (7 * k + 4 * (2 * k - 1))
    if mode == "shared":
        return common + 4 * k * (2 * tokens * groups - 1)
    if mode == "group_adaptive":
        return common + groups * 4 * k * (2 * tokens - 1)
    raise ValueError("Unknown sharing mode")
