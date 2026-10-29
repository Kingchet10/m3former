"""Frozen S5 recipes and residual recycling; no nonlinear local operations."""

import torch
import torch.nn.functional as F

# Ha, Hd, supergroup width, recycled directions, mode.
RECIPES = {
    "rrp1": (643, 224, 16, 1, "fixed"),
    "rrp2": (669, 192, 16, 2, "fixed"),
    "rrp3": (665, 192, 16, 3, "fixed"),
    "rrp2_flat": (669, 192, 16, 2, "flat"),
    "rrp2_learn": (669, 192, 16, 2, "learnable"),
}


def u_rows():
    q = 2**-0.5
    return [
        [q, q, 0.0, 0.0],
        [0.0, 0.0, q, q],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
    ]


def a_rows(r, mode):
    if r not in (1, 2, 3) or mode not in ("fixed", "learnable", "flat"):
        raise ValueError("Invalid recycling specification")
    if mode == "flat":
        if r != 2:
            raise ValueError("Flat control is 16 to 14 only")
        # Sylvester construction, natural row/column order, normalized by sqrt(16).
        h = [[1.0]]
        for _ in range(4):
            h = [row + row for row in h] + [row + [-v for v in row] for row in h]
        return [[v / 4 for v in row] for row in h[:14]]
    return [[0.5] * 4] if r == 1 else u_rows()[:r]


def project(p, u, a, mode):
    """All main groups precede all recycled supergroups in the final dimension."""
    if p.shape[-1] < 16 or p.shape[-1] % 16:
        raise ValueError("Dendritic width must contain complete supergroups")
    if mode == "flat":
        return F.linear(p.unflatten(-1, (-1, 16)), a).flatten(-2), None, None, None
    levels = F.linear(p.unflatten(-1, (-1, 4)), u)
    main, residual = levels[..., :3], levels[..., 3]
    recycled = F.linear(residual.unflatten(-1, (-1, 4)), a)
    return (
        torch.cat((main.flatten(-2), recycled.flatten(-2)), -1),
        main,
        residual,
        recycled,
    )


def output_width(ha, hd, r):
    if hd < 16 or hd % 16 or r not in (1, 2, 3):
        raise ValueError("Invalid recycling widths")
    return ha + hd // 16 * (12 + r)


def local_numel(r, mode):
    return 224 if mode == "flat" else 16 + 4 * r


def training_operations(tokens, supergroups, r, mode):
    """Direct dense contractions; counts zero entries, matching F.linear kernels.

    Includes forward, input VJP, and only learned B's parameter VJP. Shared
    parameter reduction uses the same 2*N-1 convention as previous rounds.
    """
    if min(tokens, supergroups) < 1:
        raise ValueError("Positive work required")
    if mode == "flat":
        return tokens * supergroups * (14 * (2 * 16 - 1) + 16 * (2 * 14 - 1))
    n = tokens * supergroups
    first = n * 4 * (4 * 7 + 4 * 7)
    second = n * (r * 7 + 4 * (2 * r - 1))
    return first + second + (4 * r * (2 * n - 1) if mode == "learnable" else 0)
