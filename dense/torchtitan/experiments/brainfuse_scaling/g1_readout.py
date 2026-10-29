"""Fixed normalized readout, preserving pair order within each local group."""
import math


def compress_pairs(y):
    if y.shape[-1] != 4:
        raise ValueError("G1 expects four-dimensional local groups")
    return y.unflatten(-1, (2, 2)).sum(dim=-1) / math.sqrt(2)
