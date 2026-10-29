# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license in the root LICENSE file.

"""Autograd-native rectangular output projections; no dense materialization.

Monarch uses the rectangular permutation convention in HazyResearch/fly's
blockdiag_butterfly_multiply reference. BLAST uses A_ij=U_i diag(s_ij) V_j^T.
The coupling contraction is a batched GEMM over the block axis, so its shared
parameter-gradient reduction is included in the 6 MAC-parameter/token count.
"""

import torch
import torch.nn.functional as F

KINDS = ("low_rank", "monarch", "blast")


def factor_shapes(kind, inputs, outputs, rank, blocks):
    if kind not in KINDS or min(inputs, outputs, rank, blocks) <= 0:
        raise ValueError("Invalid structured projection configuration")
    if kind == "low_rank":
        if rank > min(inputs, outputs):
            raise ValueError("Low-rank width exceeds the matrix dimensions")
        return {"right": (rank, inputs), "left": (outputs, rank)}
    if inputs % blocks or outputs % blocks:
        raise ValueError("Block projection dimensions must divide without padding")
    shapes = {
        "right": (blocks, rank, inputs // blocks),
        "left": (blocks, outputs // blocks, rank),
    }
    if kind == "monarch" and rank % blocks:
        raise ValueError("This balanced Monarch requires rank divisible by blocks")
    if kind == "blast":
        shapes["coupling"] = (blocks, blocks, rank)
    return shapes


def project(input, right, left, *, kind, coupling=None):
    if kind == "low_rank":
        return F.linear(F.linear(input, right), left)
    batch_shape = input.shape[:-1]
    blocks, rank, width = right.shape
    x = input.reshape(-1, blocks, width).transpose(0, 1)
    z = torch.bmm(x, right.transpose(1, 2)).transpose(0, 1)
    if kind == "monarch":
        # (k q) -> (r l) -> (l r), k=l=blocks and q=r=rank.
        z = z.reshape(-1, rank, blocks).transpose(1, 2)
    elif kind == "blast":
        if coupling is None:
            raise ValueError("BLAST requires block-specific coupling factors")
        # [r,i,j] @ [r,j,n], followed by output projection U_i.
        z = torch.bmm(coupling.permute(2, 0, 1), z.permute(2, 1, 0))
        z = z.permute(2, 1, 0)
    else:
        raise ValueError("Unknown structured projection")
    y = torch.bmm(z.transpose(0, 1), left.transpose(1, 2)).transpose(0, 1)
    if kind == "monarch":
        y = y.transpose(1, 2)  # Official final (l s) -> (s l) permutation.
    return y.reshape(*batch_shape, -1)


def coupling_training_flops(kind, blocks, rank, tokens):
    """Forward, input VJP and shared coupling gradient: three batched GEMMs."""
    return 6 * tokens * blocks * blocks * rank if kind == "blast" else 0
