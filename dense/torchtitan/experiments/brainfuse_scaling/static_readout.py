# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch reference algebra for the revised S-series protocol (no kernels)."""

import math

import torch


def coefficient_pairs(branches, diagonal=False):
    if branches < 1:
        raise ValueError("branches must be positive")
    return tuple(
        (i, j)
        for i in range(branches)
        for j in range(i, branches)
        if not diagonal or i == j
    )


def grouped_readout(p, coefficients=None, *, diagonal=False):
    """Return soma values, linear contribution and quadratic contribution.

    The final axis is B. Coefficients store Q's upper triangle (not the
    polynomial coefficients); off-diagonal products therefore have factor 2.
    Accumulation precision follows p; the model explicitly supplies FP32 p.
    """
    branches = p.shape[-1]
    linear = p.sum(dim=-1)
    quadratic = torch.zeros_like(linear)
    if coefficients is not None:
        pairs = coefficient_pairs(branches, diagonal)
        if coefficients.shape != (len(pairs),):
            raise ValueError("Incorrect number of independent quadratic coefficients")
        terms = []
        for k, (i, j) in enumerate(pairs):
            coefficient = coefficients[k] if i == j else 2 * coefficients[k]
            terms.append(coefficient * (p[..., i] * p[..., j]))
        quadratic = terms[0]
        for term in terms[1:]:
            quadratic = quadratic + term
        soma = (linear + quadratic) / math.sqrt(branches)
    else:
        soma = linear / math.sqrt(branches)
    return soma, linear, quadratic


def factorized_to_symmetric(u, v, weights):
    """Expand the third-version F definition; factors need not span every Q."""
    return 0.5 * (
        torch.einsum("j,ji,jk->ik", weights, u, v)
        + torch.einsum("j,ji,jk->ik", weights, v, u)
    )


def readout_operations(groups, branches, kind):
    """Count scalar arithmetic in a specified direct forward/VJP algorithm.

    groups counts all token/soma pairs in one layer and one microbatch.
    A sum of n values costs n-1 additions. The VJP starts each input gradient
    with the linear-path gradient, then adds the two product contributions.
    Shared coefficient gradients include their complete groups-1 reduction.
    Indexing, broadcasts, casts and memory traffic are not FLOPs. Autograd's
    scatter/temporary overhead is implementation dependent and not included.
    """
    if groups < 1 or kind not in ("s0", "s1_d", "s1_q"):
        raise ValueError("Invalid readout operation request")
    if kind == "s0":
        return {"forward": groups * branches, "backward": groups}
    pairs = coefficient_pairs(branches, kind == "s1_d")
    count = len(pairs)
    off_diagonal = sum(i != j for i, j in pairs)
    return {
        "forward": groups * (branches + 3 * count) + off_diagonal,
        "backward": groups * (1 + 6 * count) + count * (groups - 1) + off_diagonal,
    }
