"""Fused depth mixing for M3, second version: tiles that span several groups, and source gradients accumulated in
place.

    out[c, r, g*d + i] = sum_j w[r, c, j, g] * h_j[r, g*d + i]        (r = position, c = stream, j = source, g = group)

Tiles. A program owns BR rows and NG whole groups of d channels (or one chunk of DW channels of one wide group), so
narrow groups (d = 16 or 32) are still read in long contiguous rows, and the weight gradient of a group is summed
inside one program.

In-place accumulation. Every hidden state h_j is read by all later blocks, so in backward it receives one gradient
from each of them. Instead of returning a new gradient tensor per source and per block, which autograd then adds up,
the backward kernel adds its contribution directly to one plane of a gradient buffer shared by the whole forward
pass. `anchor` connects h_j to this buffer: it returns an empty token that the mixers take as an input, so autograd
runs the anchor's backward, which hands plane j to h_j, only after every mixer that reads h_j has added to it. The
hidden states themselves are copied into one buffer as well, so a kernel finds source j at plane j.

    token = start_pass(h0, n_states)             # buffers of hidden states and of their gradients
    token = anchor(h_j, token, j)                # once per hidden state, in order: copies h_j to plane j
    out = mix_into(token, w, streams, groups, dim, first)   # reads planes 0 .. S-1

The first mixer to run in backward (the output mixer of the last block, which reads every state) writes its planes,
the others add to them, so the buffer needs no zeroing. Each output element sums its sources in order in FP32, as in
the first version; source gradients are now rounded to BF16 once per addition instead of twice.

The buffers are module state: one forward pass with gradients at a time, followed by its backward pass (as in
training with gradient accumulation); no activation checkpointing of the blocks, no double backward. Passes without
gradients (evaluation) use the drop-in `mix` op, which reads separate tensors.
"""
import math

import torch
import triton
import triton.language as tl


def _tiles(d, groups, backward=False):
    """(NG groups per tile, DW channels per group chunk, chunks per group, BR rows, warps, pipelining stages), the
    best measured on A800 (4 x 2048 rows, mid-depth blocks of 100M-1.4B models). A fixed table, not autotuning, so
    that runs are reproducible: the weight gradient of a group is reduced in an order that depends on the tile."""
    if d & (d - 1) == 0 and d <= 32:  # narrow groups: 256 channels per row
        ng = max(1, min(triton.next_power_of_2(groups), 256 // d))
        return (ng, d, 1, 8, 2, 3) if backward else (ng, d, 1, 4, 4, 1)
    if d & (d - 1) == 0 and d <= 128:  # groups of 64: two per tile (groups of 128: one)
        ng = max(1, min(triton.next_power_of_2(groups), 128 // d))
        return ng, d, 1, 4, 2, 1
    return 1, 128, triton.cdiv(d, 128), 4, 2, 1  # wide groups (MUDDFormer): chunks of 128 channels


@triton.jit
def _fwd(H0, OFFS, W, OUT, R, D, S, G, d, n_chunks,
         C: tl.constexpr, NG: tl.constexpr, DW: tl.constexpr, BR: tl.constexpr, STAGES: tl.constexpr):
    pgc = tl.program_id(0)  # group tile x chunk, fastest: the programs of one row tile share its weights in cache
    pr = tl.program_id(1)
    gt = pgc // n_chunks
    k = pgc % n_chunks
    rows = pr * BR + tl.arange(0, BR)
    gi = gt * NG + tl.arange(0, NG)
    ch = k * DW + tl.arange(0, DW)
    rmask = rows < R
    gmask = gi < G
    rows64 = rows.to(tl.int64)
    m3 = rmask[:, None, None] & gmask[None, :, None] & (ch < d)[None, None, :]
    hoff = rows64[:, None, None] * D + (gi[None, :, None] * d + ch[None, None, :])
    wm = rmask[:, None] & gmask[None, :]
    wrow = rows64[:, None] * (C * S * G) + gi[None, :]
    acc0 = tl.zeros((BR, NG, DW), dtype=tl.float32)
    acc1 = tl.zeros((BR, NG, DW), dtype=tl.float32)
    acc2 = tl.zeros((BR, NG, DW), dtype=tl.float32)
    acc3 = tl.zeros((BR, NG, DW), dtype=tl.float32)
    for j in tl.range(0, S, num_stages=STAGES):
        off = tl.load(OFFS + j)
        h = tl.load(H0 + off + hoff, mask=m3, other=0.).to(tl.float32)
        wj = wrow + j * G
        acc0 += tl.load(W + wj, mask=wm, other=0.).to(tl.float32)[:, :, None] * h
        if C > 1:
            acc1 += tl.load(W + wj + S * G, mask=wm, other=0.).to(tl.float32)[:, :, None] * h
            acc2 += tl.load(W + wj + 2 * S * G, mask=wm, other=0.).to(tl.float32)[:, :, None] * h
            acc3 += tl.load(W + wj + 3 * S * G, mask=wm, other=0.).to(tl.float32)[:, :, None] * h
    plane = R.to(tl.int64) * D
    tl.store(OUT + hoff, acc0.to(OUT.dtype.element_ty), mask=m3)
    if C > 1:
        tl.store(OUT + plane + hoff, acc1.to(OUT.dtype.element_ty), mask=m3)
        tl.store(OUT + 2 * plane + hoff, acc2.to(OUT.dtype.element_ty), mask=m3)
        tl.store(OUT + 3 * plane + hoff, acc3.to(OUT.dtype.element_ty), mask=m3)


@triton.jit
def _bwd(H0, OFFS, W, GO, GH, GW, R, D, S, G, d, n_chunks,
         C: tl.constexpr, NG: tl.constexpr, DW: tl.constexpr, BR: tl.constexpr, ACC: tl.constexpr,
         STAGES: tl.constexpr):
    # GH: S consecutive gradient planes, written (ACC = 0) or added to (ACC = 1).
    # GW: weight gradients, layout (R, C, S, G, n_chunks): in the dtype of the weights when a group is one chunk,
    # otherwise FP32 partial sums that the caller adds up.
    pgc = tl.program_id(0)
    pr = tl.program_id(1)
    gt = pgc // n_chunks
    k = pgc % n_chunks
    rows = pr * BR + tl.arange(0, BR)
    gi = gt * NG + tl.arange(0, NG)
    ch = k * DW + tl.arange(0, DW)
    rmask = rows < R
    gmask = gi < G
    rows64 = rows.to(tl.int64)
    m3 = rmask[:, None, None] & gmask[None, :, None] & (ch < d)[None, None, :]
    hoff = rows64[:, None, None] * D + (gi[None, :, None] * d + ch[None, None, :])
    wm = rmask[:, None] & gmask[None, :]
    wrow = rows64[:, None] * (C * S * G) + gi[None, :]
    plane = R.to(tl.int64) * D
    go0 = tl.load(GO + hoff, mask=m3, other=0.).to(tl.float32)
    go1 = go0
    go2 = go0
    go3 = go0
    if C > 1:
        go1 = tl.load(GO + plane + hoff, mask=m3, other=0.).to(tl.float32)
        go2 = tl.load(GO + 2 * plane + hoff, mask=m3, other=0.).to(tl.float32)
        go3 = tl.load(GO + 3 * plane + hoff, mask=m3, other=0.).to(tl.float32)
    for j in tl.range(0, S, num_stages=STAGES):
        off = tl.load(OFFS + j)
        h = tl.load(H0 + off + hoff, mask=m3, other=0.).to(tl.float32)
        wj = wrow + j * G
        gh = tl.load(W + wj, mask=wm, other=0.).to(tl.float32)[:, :, None] * go0
        tl.store(GW + wj * n_chunks + k, tl.sum(go0 * h, axis=2).to(GW.dtype.element_ty), mask=wm)
        if C > 1:
            gh += tl.load(W + wj + S * G, mask=wm, other=0.).to(tl.float32)[:, :, None] * go1
            gh += tl.load(W + wj + 2 * S * G, mask=wm, other=0.).to(tl.float32)[:, :, None] * go2
            gh += tl.load(W + wj + 3 * S * G, mask=wm, other=0.).to(tl.float32)[:, :, None] * go3
            tl.store(GW + (wj + S * G) * n_chunks + k, tl.sum(go1 * h, axis=2).to(GW.dtype.element_ty), mask=wm)
            tl.store(GW + (wj + 2 * S * G) * n_chunks + k, tl.sum(go2 * h, axis=2).to(GW.dtype.element_ty), mask=wm)
            tl.store(GW + (wj + 3 * S * G) * n_chunks + k, tl.sum(go3 * h, axis=2).to(GW.dtype.element_ty), mask=wm)
        gp = GH + j * plane + hoff
        if ACC:
            gh += tl.load(gp, mask=m3, other=0.).to(tl.float32)
        tl.store(gp, gh.to(GH.dtype.element_ty), mask=m3)


def _offsets(hiddens):
    """Element offsets of separate source tensors from the first one, copied to the GPU without blocking the CPU."""
    base, size = hiddens[0].data_ptr(), hiddens[0].element_size()
    offs = []
    for h in hiddens:
        delta = h.data_ptr() - base
        if delta % size:
            raise ValueError('misaligned source')
        offs.append(delta // size)
    return torch.tensor(offs, dtype=torch.int64).pin_memory().to(hiddens[0].device, non_blocking=True)


_PLANES = {}


def _plane_offsets(n, plane, device):
    """Offsets of the planes of a buffer of hidden states, made on the GPU once per shape."""
    key = (n, plane, device)
    if key not in _PLANES:
        _PLANES[key] = torch.arange(n, dtype=torch.int64, device=device) * plane
    return _PLANES[key]


def _check(hiddens, w, streams, groups):
    if streams not in (1, 4):
        raise ValueError(streams)
    shape = hiddens[0].shape
    for h in hiddens:
        if h.shape != shape or h.dtype != hiddens[0].dtype or not h.is_contiguous():
            raise ValueError('sources must share shape and dtype and be contiguous')
    if w.shape[-3:] != (streams, len(hiddens), groups) or not w.is_contiguous():
        raise ValueError(('weights', tuple(w.shape)))


def _launch_forward(base, offs, n, shape, w, streams, groups):
    dim = shape[-1]
    rows = math.prod(shape) // dim
    d = dim // groups
    ng, dw, n_chunks, br, warps, stages = _tiles(d, groups)
    out = base.new_empty((streams, *shape))
    grid = (triton.cdiv(groups, ng) * n_chunks, triton.cdiv(rows, br))
    _fwd[grid](base, offs, w, out, rows, dim, n, groups, d, n_chunks, C=streams, NG=ng, DW=dw, BR=br,
               STAGES=stages, num_warps=warps)
    return out


def _launch_backward(base, offs, n, shape, w, grad, grad_planes, streams, groups, accumulate):
    """Writes (or adds) the source gradients to grad_planes (n consecutive planes); returns the weight gradient."""
    grad = grad.contiguous()
    dim = shape[-1]
    rows = math.prod(shape) // dim
    d = dim // groups
    ng, dw, n_chunks, br, warps, stages = _tiles(d, groups, backward=True)
    if n_chunks == 1:  # each group is summed inside one program: write the gradient directly
        gw = torch.empty_like(w)
    else:
        gw = torch.empty((*w.shape, n_chunks), dtype=torch.float32, device=w.device)
    grid = (triton.cdiv(groups, ng) * n_chunks, triton.cdiv(rows, br))
    _bwd[grid](base, offs, w, grad, grad_planes, gw, rows, dim, n, groups, d, n_chunks, C=streams, NG=ng, DW=dw,
               BR=br, ACC=accumulate, STAGES=stages, num_warps=warps)
    return gw if n_chunks == 1 else gw.sum(-1).to(w.dtype)


def _forward(hiddens, w, streams, groups):
    _check(hiddens, w, streams, groups)
    return _launch_forward(hiddens[0], _offsets(hiddens), len(hiddens), hiddens[0].shape, w, streams, groups)


def _backward_into(hiddens, w, grad, grad_planes, streams, groups, accumulate):
    return _launch_backward(hiddens[0], _offsets(hiddens), len(hiddens), hiddens[0].shape, w, grad, grad_planes,
                            streams, groups, accumulate)


# Drop-in replacement of mix_kernel.mix: same op, new tiles, gradients returned per source.

@torch.library.custom_op('depthmix2::mix', mutates_args=())
def mix(hiddens: list[torch.Tensor], w: torch.Tensor, streams: int, groups: int) -> torch.Tensor:
    return _forward(hiddens, w, streams, groups)


@mix.register_fake
def _(hiddens, w, streams, groups):
    return hiddens[0].new_empty((streams, *hiddens[0].shape))


@torch.library.custom_op('depthmix2::mix_backward', mutates_args=())
def mix_backward(hiddens: list[torch.Tensor], w: torch.Tensor, grad: torch.Tensor, streams: int,
                 groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    _check(hiddens, w, streams, groups)
    grad_h = hiddens[0].new_empty((len(hiddens), *hiddens[0].shape))
    return grad_h, _backward_into(hiddens, w, grad, grad_h, streams, groups, False)


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


# In-place accumulation of the source gradients. The hidden states of the current forward pass are copied into one
# buffer, so the kernels address source j as plane j (no per-call list of pointers), and their gradients go to the
# planes of a second buffer. Both buffers are module state that only the op implementations touch at run time; the ops
# are opaque to torch.compile, and the order of their side effects follows from data dependencies:
#   forward:  anchor j copies h_j into plane j and returns a token that also depends on the token of anchor j - 1, and
#             a mixer reading h_0 .. h_{S-1} takes the token of anchor S - 1;
#   backward: a mixer returns a `done` token as the gradient of its token, and anchor j passes it on to anchor j - 1,
#             so anchor j runs after every mixer that reads h_j has added to plane j, and only then hands the plane
#             to h_j as its gradient.

class _State:
    hbuf = None  # hidden states of the pass
    gbuf = None  # their gradients
    cache = None  # (hbuf, gbuf) kept for the next pass of the same shape


_STATE = _State()


def start_pass(h0, n_states):
    """Allocate the buffers of a forward pass, one plane per hidden state; the gradient buffer is not initialized
    (the first mixer in backward writes all planes). Forward passes and their backward passes must not interleave
    (gradient accumulation runs forward and backward per micro-batch); activation checkpointing of M3 blocks is not
    supported. The buffers are reused by the next pass of the same shape (allocating 2 x n_states planes per
    micro-batch fragments the allocator of a nearly full GPU). Returns the token that anchor 0 takes as its
    predecessor."""
    shape = (n_states, *h0.shape)
    c = _STATE.cache
    if c is None or c[0].shape != shape or c[0].dtype != h0.dtype or c[0].device != h0.device:
        _STATE.cache = None
        c = _STATE.cache = (h0.new_empty(shape), h0.new_empty(shape))
    _STATE.hbuf, _STATE.gbuf = c
    return h0.new_empty(0)


@torch.library.custom_op('depthmix2::anchor', mutates_args=())
def anchor(h: torch.Tensor, prev: torch.Tensor, j: int) -> torch.Tensor:
    """Copy hidden state j into the buffer; returns its token (an empty tensor). In backward, plane j of the gradient
    buffer becomes the gradient of h."""
    _STATE.hbuf[j].copy_(h)
    return h.new_empty(0)


@anchor.register_fake
def _(h, prev, j):
    return h.new_empty(0)


@torch.library.custom_op('depthmix2::anchor_backward', mutates_args=())
def anchor_backward(done: torch.Tensor, j: int, shape: list[int]) -> torch.Tensor:
    g = _STATE.gbuf[j]
    if j == 0:  # every mixer reads h_0, so its anchor is the last user of the buffers
        _STATE.hbuf = _STATE.gbuf = None
    return g


@anchor_backward.register_fake
def _(done, j, shape):
    return done.new_empty(shape)


def _anchor_setup(ctx, inputs, output):
    h, prev, j = inputs
    ctx.j, ctx.shape = j, list(h.shape)


def _anchor_backward(ctx, done):
    return anchor_backward(done, ctx.j, ctx.shape), done, None


anchor.register_autograd(_anchor_backward, setup_context=_anchor_setup)


@torch.library.custom_op('depthmix2::mix_into', mutates_args=())
def mix_into(token: torch.Tensor, w: torch.Tensor, streams: int, groups: int, dim: int, first: bool) -> torch.Tensor:
    """Mixture of the hidden states h_0 .. h_{S-1} in the buffer (S = w.shape[-2]; `token` is the token of anchor
    S - 1) whose source gradients are added to planes 0 .. S-1 of the gradient buffer. `first`: this mixer runs
    first in backward and writes the planes instead of adding."""
    hbuf = _STATE.hbuf
    n = w.shape[-2]
    if w.shape[-3:] != (streams, n, groups) or not w.is_contiguous() or n > hbuf.shape[0] or hbuf.shape[-1] != dim:
        raise ValueError(('weights', tuple(w.shape)))
    shape = hbuf.shape[1:]
    return _launch_forward(hbuf, _plane_offsets(n, shape.numel(), w.device), n, shape, w, streams, groups)


@mix_into.register_fake
def _(token, w, streams, groups, dim, first):
    return w.new_empty((streams, *w.shape[:-3], dim))


@torch.library.custom_op('depthmix2::mix_into_backward', mutates_args=())
def mix_into_backward(w: torch.Tensor, grad: torch.Tensor, streams: int, groups: int,
                      first: bool) -> tuple[torch.Tensor, torch.Tensor]:
    hbuf = _STATE.hbuf
    n = w.shape[-2]
    shape = hbuf.shape[1:]
    grad_w = _launch_backward(hbuf, _plane_offsets(n, shape.numel(), w.device), n, shape, w, grad, _STATE.gbuf,
                              streams, groups, not first)
    return grad_w, w.new_empty(0)


@mix_into_backward.register_fake
def _(w, grad, streams, groups, first):
    return torch.empty_like(w), w.new_empty(0)


def _mix_into_setup(ctx, inputs, output):
    token, w, streams, groups, dim, first = inputs
    ctx.save_for_backward(w)
    ctx.streams, ctx.groups, ctx.first = streams, groups, first


def _mix_into_backward(ctx, grad):
    w, = ctx.saved_tensors
    grad_w, done = mix_into_backward(w, grad, ctx.streams, ctx.groups, ctx.first)
    return done, grad_w, None, None, None, None


mix_into.register_autograd(_mix_into_backward, setup_context=_mix_into_setup)


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
