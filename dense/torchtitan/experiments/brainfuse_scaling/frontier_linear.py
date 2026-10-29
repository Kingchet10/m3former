"""General shared linear projection with free, orthogonal or frozen A."""

import math

import torch
import torch.nn.functional as F

# Ha, Hd, B, K, parameterization. No GA or nonlinear local path.
RECIPES = {
    **{
        f"frontier_n{n}": (853 - 11 * n, 12 * n, 4, 3, "learnable")
        for n in (10, 15, 20, 25, 30)
    },
    "frontier_b3k2": (640, 240, 3, 2, "learnable"),
    "frontier_b5k4": (638, 230, 5, 4, "learnable"),
    "frontier_b6k5": (638, 228, 6, 5, "learnable"),
    "frontier_ortho": (640, 232, 4, 3, "orthogonal"),
    "frontier_fixed": (640, 232, 4, 3, "fixed"),
}


def initial_rows(b, k):
    """Frozen analytic orthonormal rows; independent of model seed/device."""
    if not 1 <= k < b:
        raise ValueError("Require 1 <= K < B")
    if (b, k) == (4, 3):
        return [
            [2**-0.5, 2**-0.5, 0.0, 0.0],
            [0.0, 0.0, 2**-0.5, 2**-0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]
    if (b, k) == (3, 2):
        return [[2**-0.5, 2**-0.5, 0.0], [6**-0.5, -(6**-0.5), 2 * 6**-0.5]]
    # Analytic orthonormal DCT-II basis. No runtime random QR or sign ambiguity.
    return [
        [
            (
                1 / math.sqrt(b)
                if row == 0
                else math.sqrt(2 / b) * math.cos(math.pi * (col + 0.5) * row / b)
            )
            for col in range(b)
        ]
        for row in range(k)
    ]


def initial_a(b, k, *, device=None, dtype=None):
    return torch.tensor(initial_rows(b, k), device=device, dtype=dtype)


def effective_a(raw, mode):
    if raw.ndim != 2 or not 1 <= raw.shape[0] < raw.shape[1]:
        raise ValueError("A must be [K,B], 1 <= K < B")
    if mode in ("learnable", "fixed"):
        return raw
    if mode != "orthogonal":
        raise ValueError("Unknown A parameterization")
    # Reduced QR of tall [B,K]. Positive diagonal fixes the QR sign convention.
    # Full column rank is required; tests and live diagnostics monitor this.
    q, rr = torch.linalg.qr(raw.T, mode="reduced")
    sign = torch.where(rr.diagonal() < 0, -1.0, 1.0)
    return (q * sign).T


def local_linear(p, raw, mode="learnable"):
    if p.shape[-1] != raw.shape[-1]:
        raise ValueError("Input group width differs from A")
    return F.linear(p, effective_a(raw, mode))


def training_operations(tokens, groups, b, k, mode):
    """Direct local forward/input VJP/parameter VJP operation convention.

    Fixed A omits dA. QR is accounted separately as a declared leading-order
    nominal estimate, not an exact count of cuSOLVER/autograd instructions.
    """
    if (
        min(tokens, groups) < 1
        or not 1 <= k < b
        or mode not in ("learnable", "orthogonal", "fixed")
    ):
        raise ValueError("Invalid local dimensions or mode")
    common = tokens * groups * (k * (2 * b - 1) + b * (2 * k - 1))
    return (
        common
        + (0 if mode == "fixed" else b * k * (2 * tokens * groups - 1))
        + orthogonalization_estimate(b, k, mode)
    )


def orthogonalization_estimate(b, k, mode):
    # Leading-order Householder QR forward plus QR VJP/triangular solve work.
    # Fixed convention 6*B*K^2 + 4*K^3 per matrix, independent of token count.
    return 6 * b * k * k + 4 * k * k * k if mode == "orthogonal" else 0
