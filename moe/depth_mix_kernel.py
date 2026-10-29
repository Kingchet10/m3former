"""Fused depth mixing for depth dendrites.

    out[c, r, g*d + i] = sum_j w[r, c, j, g] * h_j[r, g*d + i]        (r = position, c = stream, j = source, g = group)

Forward reads every source once and writes all streams; backward reads the stream gradients and every source once,
writes the source gradients and the weight gradients (summed over each group's channels). The sources are separate
tensors of identical shape; the kernels address them by element offsets from the first one, so nothing is stacked.
Wrapped as torch.library custom ops so that per-block torch.compile treats the mixing as one opaque node.
"""
import os

import torch
import triton
import triton.language as tl


# Program order: group-major (default) launches the G x chunks programs of one row tile back to back, so the per-row
# mixing weights they all read stay in cache; row-major (SA_DEPTH_GRID=row) is the original order. Every output
# element is computed by the same arithmetic in both orders, so results are bitwise identical.
GROUP_MAJOR = os.environ.get('SA_DEPTH_GRID', 'group') != 'row'


def _grid(rows, br, groups, n_chunks):
    """(grid, group-major flag). With one group there is nothing to share, so the original order is kept (measured:
    group-major is 1.5-5x faster for 10-40 groups and no faster for one group)."""
    tiles = triton.cdiv(rows, br)
    gm = GROUP_MAJOR and groups > 1
    return ((groups * n_chunks, tiles) if gm else (tiles, groups * n_chunks)), gm


def _launch_cfg(d):
    """(channel tile, chunks per group, row tile, warps); best measured on A800 (bf16, 4x2048 rows) for
    D 640 / 1024 with 1 or 16 groups: wide groups CH 128-256, narrow groups CH 64 with 8 rows."""
    ch = min(triton.next_power_of_2(d), 256 if d >= 1024 else 128)
    br = 16 if ch >= 256 else 8
    return ch, triton.cdiv(d, ch), br, 4


@triton.jit
def _mix_fwd(H0, OFFS, W, OUT, R, D, S, d, n_chunks,
             C: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr,
             GM: tl.constexpr):
    if GM:  # group-major grid: programs for the same row tile run back to back and share the weights in cache
        pgc = tl.program_id(0)
        pr = tl.program_id(1)
    else:
        pr = tl.program_id(0)
        pgc = tl.program_id(1)
    g = pgc // n_chunks
    k = pgc % n_chunks
    rows = pr * BR + tl.arange(0, BR)
    cols = k * CH + tl.arange(0, CH)
    rmask = rows < R
    cmask = cols < d
    m2 = rmask[:, None] & cmask[None, :]
    rows64 = rows.to(tl.int64)
    hoff = rows64[:, None] * D + (g * d + cols)[None, :]
    wrow = rows64 * (C * S * G) + g
    acc0 = tl.zeros((BR, CH), dtype=tl.float32)
    acc1 = tl.zeros((BR, CH), dtype=tl.float32)
    acc2 = tl.zeros((BR, CH), dtype=tl.float32)
    acc3 = tl.zeros((BR, CH), dtype=tl.float32)
    for j in range(S):
        off = tl.load(OFFS + j)
        h = tl.load(H0 + off + hoff, mask=m2, other=0.).to(tl.float32)
        wj = wrow + j * G
        acc0 += tl.load(W + wj, mask=rmask, other=0.).to(tl.float32)[:, None] * h
        if C > 1:
            acc1 += tl.load(W + wj + S * G, mask=rmask, other=0.).to(tl.float32)[:, None] * h
            acc2 += tl.load(W + wj + 2 * S * G, mask=rmask, other=0.).to(tl.float32)[:, None] * h
            acc3 += tl.load(W + wj + 3 * S * G, mask=rmask, other=0.).to(tl.float32)[:, None] * h
    plane = R.to(tl.int64) * D
    tl.store(OUT + hoff, acc0.to(OUT.dtype.element_ty), mask=m2)
    if C > 1:
        tl.store(OUT + plane + hoff, acc1.to(OUT.dtype.element_ty), mask=m2)
        tl.store(OUT + 2 * plane + hoff, acc2.to(OUT.dtype.element_ty), mask=m2)
        tl.store(OUT + 3 * plane + hoff, acc3.to(OUT.dtype.element_ty), mask=m2)


@triton.jit
def _mix_bwd(H0, OFFS, W, GO, GH, GW, R, D, S, d, n_chunks,
             C: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr,
             GM: tl.constexpr):
    # GW: partial weight gradients, layout (R, C, S, G, n_chunks); summed over chunks by the caller
    if GM:  # group-major grid: programs for the same row tile run back to back and share the weights in cache
        pgc = tl.program_id(0)
        pr = tl.program_id(1)
    else:
        pr = tl.program_id(0)
        pgc = tl.program_id(1)
    g = pgc // n_chunks
    k = pgc % n_chunks
    rows = pr * BR + tl.arange(0, BR)
    cols = k * CH + tl.arange(0, CH)
    rmask = rows < R
    cmask = cols < d
    m2 = rmask[:, None] & cmask[None, :]
    rows64 = rows.to(tl.int64)
    hoff = rows64[:, None] * D + (g * d + cols)[None, :]
    wrow = rows64 * (C * S * G) + g
    plane = R.to(tl.int64) * D
    go0 = tl.load(GO + hoff, mask=m2, other=0.).to(tl.float32)
    go1 = go0
    go2 = go0
    go3 = go0
    if C > 1:
        go1 = tl.load(GO + plane + hoff, mask=m2, other=0.).to(tl.float32)
        go2 = tl.load(GO + 2 * plane + hoff, mask=m2, other=0.).to(tl.float32)
        go3 = tl.load(GO + 3 * plane + hoff, mask=m2, other=0.).to(tl.float32)
    for j in range(S):
        off = tl.load(OFFS + j)
        h = tl.load(H0 + off + hoff, mask=m2, other=0.).to(tl.float32)
        wj = wrow + j * G
        w0 = tl.load(W + wj, mask=rmask, other=0.).to(tl.float32)
        gh = w0[:, None] * go0
        tl.store(GW + wj * n_chunks + k, tl.sum(go0 * h, axis=1), mask=rmask)
        if C > 1:
            w1 = tl.load(W + wj + S * G, mask=rmask, other=0.).to(tl.float32)
            w2 = tl.load(W + wj + 2 * S * G, mask=rmask, other=0.).to(tl.float32)
            w3 = tl.load(W + wj + 3 * S * G, mask=rmask, other=0.).to(tl.float32)
            gh += w1[:, None] * go1 + w2[:, None] * go2 + w3[:, None] * go3
            tl.store(GW + (wj + S * G) * n_chunks + k, tl.sum(go1 * h, axis=1), mask=rmask)
            tl.store(GW + (wj + 2 * S * G) * n_chunks + k, tl.sum(go2 * h, axis=1), mask=rmask)
            tl.store(GW + (wj + 3 * S * G) * n_chunks + k, tl.sum(go3 * h, axis=1), mask=rmask)
        tl.store(GH + j * plane + hoff, gh.to(GH.dtype.element_ty), mask=m2)


def _offsets(hiddens):
    base, size = hiddens[0].data_ptr(), hiddens[0].element_size()
    offs = []
    for h in hiddens:
        delta = h.data_ptr() - base
        if delta % size:
            raise ValueError('misaligned source')
        offs.append(delta // size)
    return torch.tensor(offs, dtype=torch.int64, device=hiddens[0].device)


def _check(hiddens, w, streams, groups):
    if streams not in (1, 4):
        raise ValueError(streams)
    shape = hiddens[0].shape
    for h in hiddens:
        if h.shape != shape or h.dtype != hiddens[0].dtype or not h.is_contiguous():
            raise ValueError('sources must share shape and dtype and be contiguous')
    if w.shape[-3:] != (streams, len(hiddens), groups) or not w.is_contiguous():
        raise ValueError(('weights', tuple(w.shape)))


@torch.library.custom_op('depthmix::mix', mutates_args=())
def mix(hiddens: list[torch.Tensor], w: torch.Tensor, streams: int, groups: int) -> torch.Tensor:
    _check(hiddens, w, streams, groups)
    shape = hiddens[0].shape
    dim = shape[-1]
    rows = hiddens[0].numel() // dim
    d = dim // groups
    out = hiddens[0].new_empty((streams, *shape))
    ch, n_chunks, br, warps = _launch_cfg(d)
    grid, gm = _grid(rows, br, groups, n_chunks)
    _mix_fwd[grid](hiddens[0], _offsets(hiddens), w, out, rows, dim, len(hiddens), d, n_chunks,
                   C=streams, G=groups, CH=ch, BR=br, GM=gm, num_warps=warps)
    return out


@mix.register_fake
def _(hiddens, w, streams, groups):
    return hiddens[0].new_empty((streams, *hiddens[0].shape))


@torch.library.custom_op('depthmix::mix_backward', mutates_args=())
def mix_backward(hiddens: list[torch.Tensor], w: torch.Tensor, grad: torch.Tensor, streams: int,
                 groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    _check(hiddens, w, streams, groups)
    grad = grad.contiguous()
    shape = hiddens[0].shape
    dim = shape[-1]
    rows = hiddens[0].numel() // dim
    d = dim // groups
    grad_h = hiddens[0].new_empty((len(hiddens), *shape))
    ch, n_chunks, br, warps = _launch_cfg(d)
    partial = torch.empty((*w.shape, n_chunks), dtype=torch.float32, device=w.device)
    grid, gm = _grid(rows, br, groups, n_chunks)
    _mix_bwd[grid](hiddens[0], _offsets(hiddens), w, grad, grad_h, partial, rows, dim, len(hiddens), d, n_chunks,
                   C=streams, G=groups, CH=ch, BR=br, GM=gm, num_warps=warps)
    return grad_h, partial.sum(-1).to(w.dtype)


@mix_backward.register_fake
def _(hiddens, w, grad, streams, groups):
    return hiddens[0].new_empty((len(hiddens), *hiddens[0].shape)), torch.empty_like(w)


def _setup(ctx, inputs, output):
    hiddens, w, streams, groups = inputs
    ctx.save_for_backward(w, *hiddens)
    ctx.streams, ctx.groups = streams, groups


def _backward(ctx, grad):
    w, *hiddens = ctx.saved_tensors
    grad_h, grad_w = mix_backward(list(hiddens), w, grad, ctx.streams, ctx.groups)
    return list(grad_h.unbind(0)), grad_w, None, None


mix.register_autograd(_backward, setup_context=_setup)


def mix_reference(hiddens, w, streams, groups):
    """The plain PyTorch definition, for tests."""
    b, t, dim = hiddens[0].shape
    out = []
    for c in range(streams):
        acc = None
        for j, h in enumerate(hiddens):
            term = w[:, :, c, j, :, None] * h.view(b, t, groups, dim // groups)
            acc = term if acc is None else acc + term
        out.append(acc.reshape(b, t, dim))
    return torch.stack(out)
