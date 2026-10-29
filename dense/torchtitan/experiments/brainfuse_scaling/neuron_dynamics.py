"""Stage A: per-neuron linear membrane dynamics with real / complex modes.

Each neuron integrates its somatic current with K modes
    h_t = lam * h_{t-1} + (1 - |lam|) * x_t,   lam = exp(-exp(nu)) * exp(i*theta),
reset at document starts, and adds the readout Re(sum_k c_k h_k) to the current
before the somatic nonlinearity (placement 'gate'), to the analog path ('up'),
or after the nonlinearity ('post', the Step 1 T4 position). c is zero at
initialisation, so every variant is exactly the SwiGLU baseline at step 0.
"""
from dataclasses import dataclass, fields, replace
import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from torchtitan.models.common.feed_forward import FeedForward

from .neuron_complexity import TemporalBlock

MODES = {'m1': (1, False), 'c1': (1, True), 'c2': (2, True)}  # (K, complex)
PLACEMENTS = ('gate', 'up', 'post')
WINDOWS = (0, 4, 8, 16, 32, 64, 128)
_GROUPS = 16
_C, _F = 64, 16


def local_parameter_count(dynamics, hidden):
    k, cplx = MODES[dynamics]
    return hidden * k * (4 if cplx else 2)


def local_operations(dynamics, layers, hidden, physical_tokens):
    k, cplx = MODES[dynamics]
    return 3 * layers * physical_tokens * hidden * k * (12 if cplx else 5)


def modes_reference(x, lam_re, lam_im, gain, reset):
    """Sequential differentiable reference. x [B,T,H]; lam/gain [H,K]; reset [B,T]."""
    b, t, h = x.shape
    sr = x.new_zeros(b, h, lam_re.shape[-1])
    si = torch.zeros_like(sr)
    out_r, out_i = [], []
    for step in range(t):
        keep = (~reset[:, step]).to(x.dtype)[:, None, None]
        pr, pi = sr * keep, si * keep
        sr = pr * lam_re - pi * lam_im + gain * x[:, step, :, None]
        si = pr * lam_im + pi * lam_re
        out_r.append(sr)
        out_i.append(si)
    return torch.stack(out_r, 1), torch.stack(out_i, 1)


try:
    import triton as tr
    import triton.language as tl

    @tr.jit
    def _ccompose(a0r, a0i, b0r, b0i, a1r, a1i, b1r, b1i):
        return (a0r * a1r - a0i * a1i, a0r * a1i + a0i * a1r,
                b0r * a1r - b0i * a1i + b1r, b0r * a1i + b0i * a1r + b1i)

    @tr.jit
    def _mfwd(X, LR, LI, G, RESET, SR, SI, T: tl.constexpr, H: tl.constexpr,
              K: tl.constexpr, C: tl.constexpr, F: tl.constexpr):
        batch = tl.program_id(0)
        j = tl.program_id(1) * F + tl.arange(0, F)
        jm = j < H * K
        lr = tl.load(LR + j, jm, 0.)
        li = tl.load(LI + j, jm, 0.)
        g = tl.load(G + j, jm, 0.)
        pr = tl.full((F,), 0, tl.float32)
        pi = tl.full((F,), 0, tl.float32)
        rows = tl.arange(0, C)
        for start in range(tr.cdiv(T, C)):
            t = start * C + rows
            reset = tl.load(RESET + batch * T + t, t < T, 1)
            m2 = (t[:, None] < T) & jm[None, :]
            x = tl.load(X + (batch * T + t[:, None]) * H + j[None, :] // K, m2, 0.)
            live = (t[:, None] < T) & (~reset[:, None])
            ar = tl.where(live, lr[None, :], 0.)
            ai = tl.where(live, li[None, :], 0.)
            br = g[None, :] * x
            bi = br * 0.
            aar, aai, bbr, bbi = tl.associative_scan((ar, ai, br, bi), 0, _ccompose)
            sr = pr[None, :] * aar - pi[None, :] * aai + bbr
            si = pr[None, :] * aai + pi[None, :] * aar + bbi
            off = (batch * T + t[:, None]) * H * K + j[None, :]
            tl.store(SR + off, sr, m2)
            tl.store(SI + off, si, m2)
            last = rows[:, None] == min(C, T - start * C) - 1
            pr = tl.sum(tl.where(last, sr, 0.), 0)
            pi = tl.sum(tl.where(last, si, 0.), 0)

    @tr.jit
    def _mbwd(X, LR, LI, G, RESET, SR, SI, DSR, DSI, DX, DLR, DLI, DG,
              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
              C: tl.constexpr, F: tl.constexpr):
        batch = tl.program_id(0)
        j = tl.program_id(1) * F + tl.arange(0, F)
        jm = j < H * K
        lr = tl.load(LR + j, jm, 0.)
        li = tl.load(LI + j, jm, 0.)
        g = tl.load(G + j, jm, 0.)
        cr = tl.full((F,), 0, tl.float32)
        ci = tl.full((F,), 0, tl.float32)
        acc_r = tl.full((F,), 0, tl.float32)
        acc_i = tl.full((F,), 0, tl.float32)
        acc_g = tl.full((F,), 0, tl.float32)
        rows = tl.arange(0, C)
        for start in range(tr.cdiv(T, C)):
            t = T - 1 - (start * C + rows)
            valid = t >= 0
            m2 = valid[:, None] & jm[None, :]
            reset_next = tl.load(RESET + batch * T + t + 1, valid & (t + 1 < T), 1)
            live = valid[:, None] & (~reset_next[:, None])
            ar = tl.where(live, lr[None, :], 0.)
            ai = tl.where(live, -li[None, :], 0.)
            off = (batch * T + t[:, None]) * H * K + j[None, :]
            dr = tl.load(DSR + off, m2, 0.)
            di = tl.load(DSI + off, m2, 0.)
            aar, aai, bbr, bbi = tl.associative_scan((ar, ai, dr, di), 0, _ccompose)
            adr = cr[None, :] * aar - ci[None, :] * aai + bbr
            adi = cr[None, :] * aai + ci[None, :] * aar + bbi
            reset = tl.load(RESET + batch * T + t, valid, 1)
            pm = m2 & (t[:, None] > 0) & (~reset[:, None])
            pr = tl.load(SR + off - H * K, pm, 0.)
            pi = tl.load(SI + off - H * K, pm, 0.)
            x = tl.load(X + (batch * T + t[:, None]) * H + j[None, :] // K, m2, 0.)
            tl.store(DX + off, g[None, :] * adr, m2)
            acc_r += tl.sum(adr * pr + adi * pi, 0)
            acc_i += tl.sum(adi * pr - adr * pi, 0)
            acc_g += tl.sum(adr * x, 0)
            last = rows[:, None] == min(C, T - start * C) - 1
            cr = tl.sum(tl.where(last, adr, 0.), 0)
            ci = tl.sum(tl.where(last, adi, 0.), 0)
        tl.store(DLR + batch * H * K + j, acc_r, jm)
        tl.store(DLI + batch * H * K + j, acc_i, jm)
        tl.store(DG + batch * H * K + j, acc_g, jm)

    class _Modes(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam_re, lam_im, gain, reset):
            x = x.contiguous()
            lam_re, lam_im, gain = lam_re.contiguous(), lam_im.contiguous(), gain.contiguous()
            reset = reset.contiguous()
            b, t, h = x.shape
            k = lam_re.shape[-1]
            sr = torch.empty((b, t, h, k), device=x.device, dtype=torch.float32)
            si = torch.empty_like(sr)
            _mfwd[(b, tr.cdiv(h * k, _F))](x, lam_re, lam_im, gain, reset, sr, si,
                                           t, h, k, _C, _F, enable_fp_fusion=False)
            ctx.save_for_backward(x, lam_re, lam_im, gain, reset, sr, si)
            return sr, si

        @staticmethod
        def backward(ctx, dsr, dsi):
            x, lam_re, lam_im, gain, reset, sr, si = ctx.saved_tensors
            b, t, h = x.shape
            k = lam_re.shape[-1]
            dsr = torch.zeros_like(sr) if dsr is None else dsr.float().contiguous()
            dsi = torch.zeros_like(si) if dsi is None else dsi.float().contiguous()
            dx = torch.empty_like(sr)
            dlr = torch.empty((b, h * k), device=x.device, dtype=torch.float32)
            dli, dg = torch.empty_like(dlr), torch.empty_like(dlr)
            _mbwd[(b, tr.cdiv(h * k, _F))](x, lam_re, lam_im, gain, reset, sr, si, dsr, dsi,
                                           dx, dlr, dli, dg, t, h, k, _C, _F,
                                           enable_fp_fusion=False)
            return (dx.sum(-1), dlr.sum(0).view(h, k), dli.sum(0).view(h, k),
                    dg.sum(0).view(h, k), None)

    modes_triton = _Modes.apply

    @tr.jit
    def _ffwd(X, LR, LI, G, CR, CI, RESET, SR, SI, OUT, T: tl.constexpr, H: tl.constexpr,
              K: tl.constexpr, C: tl.constexpr, F: tl.constexpr):
        batch = tl.program_id(0)
        n = tl.program_id(1) * F + tl.arange(0, F)
        nm = n < H
        rows = tl.arange(0, C)
        for k in tl.static_range(K):
            lr = tl.load(LR + n * K + k, nm, 0.)
            li = tl.load(LI + n * K + k, nm, 0.)
            g = tl.load(G + n * K + k, nm, 0.)
            cr = tl.load(CR + n * K + k, nm, 0.)
            ci = tl.load(CI + n * K + k, nm, 0.)
            pr = tl.full((F,), 0, tl.float32)
            pi = tl.full((F,), 0, tl.float32)
            for start in range(tr.cdiv(T, C)):
                t = start * C + rows
                reset = tl.load(RESET + batch * T + t, t < T, 1)
                m2 = (t[:, None] < T) & nm[None, :]
                x = tl.load(X + (batch * T + t[:, None]) * H + n[None, :], m2, 0.)
                live = (t[:, None] < T) & (~reset[:, None])
                ar = tl.where(live, lr[None, :], 0.)
                ai = tl.where(live, li[None, :], 0.)
                br = g[None, :] * x
                bi = br * 0.
                aar, aai, bbr, bbi = tl.associative_scan((ar, ai, br, bi), 0, _ccompose)
                sr = pr[None, :] * aar - pi[None, :] * aai + bbr
                si = pr[None, :] * aai + pi[None, :] * aar + bbi
                off = ((batch * T + t[:, None]) * H + n[None, :]) * K + k
                tl.store(SR + off, sr, m2)
                tl.store(SI + off, si, m2)
                o = OUT + (batch * T + t[:, None]) * H + n[None, :]
                contrib = cr[None, :] * sr - ci[None, :] * si
                if k == 0:
                    tl.store(o, contrib, m2)
                else:
                    tl.store(o, tl.load(o, m2, 0.) + contrib, m2)
                last = rows[:, None] == min(C, T - start * C) - 1
                pr = tl.sum(tl.where(last, sr, 0.), 0)
                pi = tl.sum(tl.where(last, si, 0.), 0)

    @tr.jit
    def _fbwd(X, LR, LI, G, CR, CI, RESET, SR, SI, DOUT, DX, DLR, DLI, DG, DCR, DCI,
              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, C: tl.constexpr, F: tl.constexpr):
        batch = tl.program_id(0)
        n = tl.program_id(1) * F + tl.arange(0, F)
        nm = n < H
        rows = tl.arange(0, C)
        for k in tl.static_range(K):
            lr = tl.load(LR + n * K + k, nm, 0.)
            li = tl.load(LI + n * K + k, nm, 0.)
            g = tl.load(G + n * K + k, nm, 0.)
            cr = tl.load(CR + n * K + k, nm, 0.)
            ci = tl.load(CI + n * K + k, nm, 0.)
            car = tl.full((F,), 0, tl.float32)
            cai = tl.full((F,), 0, tl.float32)
            acc_r = tl.full((F,), 0, tl.float32)
            acc_i = tl.full((F,), 0, tl.float32)
            acc_g = tl.full((F,), 0, tl.float32)
            acc_cr = tl.full((F,), 0, tl.float32)
            acc_ci = tl.full((F,), 0, tl.float32)
            for start in range(tr.cdiv(T, C)):
                t = T - 1 - (start * C + rows)
                valid = t >= 0
                m2 = valid[:, None] & nm[None, :]
                reset_next = tl.load(RESET + batch * T + t + 1, valid & (t + 1 < T), 1)
                live = valid[:, None] & (~reset_next[:, None])
                ar = tl.where(live, lr[None, :], 0.)
                ai = tl.where(live, -li[None, :], 0.)
                base = (batch * T + t[:, None]) * H + n[None, :]
                dout = tl.load(DOUT + base, m2, 0.)
                dr = dout * cr[None, :]
                di = -dout * ci[None, :]
                aar, aai, bbr, bbi = tl.associative_scan((ar, ai, dr, di), 0, _ccompose)
                adr = car[None, :] * aar - cai[None, :] * aai + bbr
                adi = car[None, :] * aai + cai[None, :] * aar + bbi
                off = base * K + k
                sr = tl.load(SR + off, m2, 0.)
                si = tl.load(SI + off, m2, 0.)
                reset = tl.load(RESET + batch * T + t, valid, 1)
                pm = m2 & (t[:, None] > 0) & (~reset[:, None])
                pr = tl.load(SR + off - H * K, pm, 0.)
                pi = tl.load(SI + off - H * K, pm, 0.)
                x = tl.load(X + base, m2, 0.)
                dxo = DX + base
                if k == 0:
                    tl.store(dxo, g[None, :] * adr, m2)
                else:
                    tl.store(dxo, tl.load(dxo, m2, 0.) + g[None, :] * adr, m2)
                acc_r += tl.sum(adr * pr + adi * pi, 0)
                acc_i += tl.sum(adi * pr - adr * pi, 0)
                acc_g += tl.sum(adr * x, 0)
                acc_cr += tl.sum(dout * sr, 0)
                acc_ci -= tl.sum(dout * si, 0)
                last = rows[:, None] == min(C, T - start * C) - 1
                car = tl.sum(tl.where(last, adr, 0.), 0)
                cai = tl.sum(tl.where(last, adi, 0.), 0)
            o = batch * H * K + n * K + k
            tl.store(DLR + o, acc_r, nm)
            tl.store(DLI + o, acc_i, nm)
            tl.store(DG + o, acc_g, nm)
            tl.store(DCR + o, acc_cr, nm)
            tl.store(DCI + o, acc_ci, nm)

    _FC, _FF = 64, 64

    class _Fused(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam_re, lam_im, gain, c_re, c_im, reset):
            x = x.contiguous()
            params = [v.contiguous() for v in (lam_re, lam_im, gain, c_re, c_im)]
            reset = reset.contiguous()
            b, t, h = x.shape
            k = lam_re.shape[-1]
            sr = torch.empty((b, t, h, k), device=x.device, dtype=torch.float32)
            si = torch.empty_like(sr)
            out = torch.empty((b, t, h), device=x.device, dtype=torch.float32)
            _ffwd[(b, tr.cdiv(h, _FF))](x, *params, reset, sr, si, out, t, h, k, _FC, _FF)
            ctx.save_for_backward(x, *params, reset, sr, si)
            return out

        @staticmethod
        def backward(ctx, dout):
            x, lam_re, lam_im, gain, c_re, c_im, reset, sr, si = ctx.saved_tensors
            b, t, h = x.shape
            k = lam_re.shape[-1]
            dx = torch.empty_like(x)
            grads = [torch.empty((b, h * k), device=x.device, dtype=torch.float32) for _ in range(5)]
            _fbwd[(b, tr.cdiv(h, _FF))](x, lam_re, lam_im, gain, c_re, c_im, reset, sr, si,
                                        dout.float().contiguous(), dx, *grads, t, h, k, _FC, _FF)
            dlr, dli, dg, dcr, dci = [v.sum(0).view(h, k) for v in grads]
            return dx, dlr, dli, dg, dcr, dci, None

    fused_triton = _Fused.apply

    @tr.jit
    def _sfwd(X, LR, LI, G, CR, CI, RESET, SR, SI, OUT, T: tl.constexpr, H: tl.constexpr,
              K: tl.constexpr, F: tl.constexpr):
        batch = tl.program_id(0)
        n = tl.program_id(1) * F + tl.arange(0, F)
        nm = n < H
        l0r = tl.load(LR + n * K, nm, 0.); l0i = tl.load(LI + n * K, nm, 0.); g0 = tl.load(G + n * K, nm, 0.)
        c0r = tl.load(CR + n * K, nm, 0.); c0i = tl.load(CI + n * K, nm, 0.)
        l1r = tl.load(LR + n * K + 1, nm & (K == 2), 0.); l1i = tl.load(LI + n * K + 1, nm & (K == 2), 0.)
        g1 = tl.load(G + n * K + 1, nm & (K == 2), 0.)
        c1r = tl.load(CR + n * K + 1, nm & (K == 2), 0.); c1i = tl.load(CI + n * K + 1, nm & (K == 2), 0.)
        p0r = tl.full((F,), 0, tl.float32); p0i = tl.full((F,), 0, tl.float32)
        p1r = tl.full((F,), 0, tl.float32); p1i = tl.full((F,), 0, tl.float32)
        for t in range(T):
            keep = tl.load(RESET + batch * T + t) == 0
            row = (batch * T + t) * H + n
            x = tl.load(X + row, nm, 0.)
            q0r = tl.where(keep, p0r, 0.); q0i = tl.where(keep, p0i, 0.)
            p0r = l0r * q0r - l0i * q0i + g0 * x
            p0i = l0r * q0i + l0i * q0r
            tl.store(SR + row * K, p0r, nm); tl.store(SI + row * K, p0i, nm)
            y = c0r * p0r - c0i * p0i
            if K == 2:
                q1r = tl.where(keep, p1r, 0.); q1i = tl.where(keep, p1i, 0.)
                p1r = l1r * q1r - l1i * q1i + g1 * x
                p1i = l1r * q1i + l1i * q1r
                tl.store(SR + row * K + 1, p1r, nm); tl.store(SI + row * K + 1, p1i, nm)
                y += c1r * p1r - c1i * p1i
            tl.store(OUT + row, y, nm)

    @tr.jit
    def _sbwd(X, LR, LI, G, CR, CI, RESET, SR, SI, DOUT, DX, DLR, DLI, DG, DCR, DCI,
              T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, F: tl.constexpr):
        batch = tl.program_id(0)
        n = tl.program_id(1) * F + tl.arange(0, F)
        nm = n < H
        m1 = nm & (K == 2)
        l0r = tl.load(LR + n * K, nm, 0.); l0i = tl.load(LI + n * K, nm, 0.); g0 = tl.load(G + n * K, nm, 0.)
        c0r = tl.load(CR + n * K, nm, 0.); c0i = tl.load(CI + n * K, nm, 0.)
        l1r = tl.load(LR + n * K + 1, m1, 0.); l1i = tl.load(LI + n * K + 1, m1, 0.); g1 = tl.load(G + n * K + 1, m1, 0.)
        c1r = tl.load(CR + n * K + 1, m1, 0.); c1i = tl.load(CI + n * K + 1, m1, 0.)
        z = tl.full((F,), 0, tl.float32)
        a0r, a0i, a1r, a1i = z, z, z, z
        d0lr, d0li, d0g, d0cr, d0ci = z, z, z, z, z
        d1lr, d1li, d1g, d1cr, d1ci = z, z, z, z, z
        last = (batch * T + T - 1) * H + n
        s0r = tl.load(SR + last * K, nm, 0.); s0i = tl.load(SI + last * K, nm, 0.)
        s1r = tl.load(SR + last * K + 1, m1, 0.); s1i = tl.load(SI + last * K + 1, m1, 0.)
        keep_next = z == 1.  # all False: nothing after the final token
        for i in range(T):
            t = T - 1 - i
            row = (batch * T + t) * H + n
            keep = tl.load(RESET + batch * T + t) == 0
            has_prev = keep & (t > 0)
            prow = row - H
            dout = tl.load(DOUT + row, nm, 0.)
            x = tl.load(X + row, nm, 0.)
            r0r = tl.load(SR + prow * K, nm & (t > 0), 0.); r0i = tl.load(SI + prow * K, nm & (t > 0), 0.)
            q0r = tl.where(has_prev, r0r, 0.); q0i = tl.where(has_prev, r0i, 0.)
            b0r = tl.where(keep_next, a0r, 0.); b0i = tl.where(keep_next, a0i, 0.)
            a0r = dout * c0r + l0r * b0r + l0i * b0i
            a0i = -dout * c0i + l0r * b0i - l0i * b0r
            d0lr += a0r * q0r + a0i * q0i
            d0li += a0i * q0r - a0r * q0i
            d0g += a0r * x
            d0cr += dout * s0r
            d0ci -= dout * s0i
            dx = g0 * a0r
            s0r, s0i = r0r, r0i
            if K == 2:
                r1r = tl.load(SR + prow * K + 1, m1 & (t > 0), 0.); r1i = tl.load(SI + prow * K + 1, m1 & (t > 0), 0.)
                q1r = tl.where(has_prev, r1r, 0.); q1i = tl.where(has_prev, r1i, 0.)
                b1r = tl.where(keep_next, a1r, 0.); b1i = tl.where(keep_next, a1i, 0.)
                a1r = dout * c1r + l1r * b1r + l1i * b1i
                a1i = -dout * c1i + l1r * b1i - l1i * b1r
                d1lr += a1r * q1r + a1i * q1i
                d1li += a1i * q1r - a1r * q1i
                d1g += a1r * x
                d1cr += dout * s1r
                d1ci -= dout * s1i
                dx += g1 * a1r
                s1r, s1i = r1r, r1i
            tl.store(DX + row, dx, nm)
            keep_next = keep | (z == 1.)
        o = batch * H * K + n * K
        tl.store(DLR + o, d0lr, nm); tl.store(DLI + o, d0li, nm); tl.store(DG + o, d0g, nm)
        tl.store(DCR + o, d0cr, nm); tl.store(DCI + o, d0ci, nm)
        tl.store(DLR + o + 1, d1lr, m1); tl.store(DLI + o + 1, d1li, m1); tl.store(DG + o + 1, d1g, m1)
        tl.store(DCR + o + 1, d1cr, m1); tl.store(DCI + o + 1, d1ci, m1)

    _SF = 128

    class _Seq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam_re, lam_im, gain, c_re, c_im, reset):
            x = x.contiguous()
            params = [v.contiguous() for v in (lam_re, lam_im, gain, c_re, c_im)]
            reset = reset.contiguous()
            b, t, h = x.shape
            k = lam_re.shape[-1]
            if k not in (1, 2):
                raise ValueError('sequential kernel supports one or two modes')
            sr = torch.empty((b, t, h, k), device=x.device, dtype=torch.float32)
            si = torch.empty_like(sr)
            out = torch.empty((b, t, h), device=x.device, dtype=torch.float32)
            _sfwd[(b, tr.cdiv(h, _SF))](x, *params, reset, sr, si, out, t, h, k, _SF)
            ctx.save_for_backward(x, *params, reset, sr, si)
            return out

        @staticmethod
        def backward(ctx, dout):
            x, lam_re, lam_im, gain, c_re, c_im, reset, sr, si = ctx.saved_tensors
            b, t, h = x.shape
            k = lam_re.shape[-1]
            dx = torch.empty_like(x)
            grads = [torch.empty((b, h * k), device=x.device, dtype=torch.float32) for _ in range(5)]
            _sbwd[(b, tr.cdiv(h, _SF))](x, lam_re, lam_im, gain, c_re, c_im, reset, sr, si,
                                        dout.float().contiguous(), dx, *grads, t, h, k, _SF)
            dlr, dli, dg, dcr, dci = [v.sum(0).view(h, k) for v in grads]
            return dx, dlr, dli, dg, dcr, dci, None

    sequential_triton = _Seq.apply
except ImportError:  # pragma: no cover - CPU-only environments
    modes_triton = None
    fused_triton = None
    sequential_triton = None


def modes_scan(x, lam_re, lam_im, gain, reset, backend='auto'):
    if backend == 'triton' or (backend == 'auto' and x.is_cuda and modes_triton is not None):
        return modes_triton(x, lam_re, lam_im, gain, reset)
    return modes_reference(x, lam_re, lam_im, gain, reset)


def modes_readout(x, lam_re, lam_im, gain, c_re, c_im, reset, backend='auto'):
    """Re(sum_k c_k h_k) for every neuron; fused Triton kernel on CUDA."""
    if backend == 'sequential' or (backend == 'auto' and x.is_cuda and sequential_triton is not None):
        return sequential_triton(x, lam_re, lam_im, gain, c_re, c_im, reset)
    if backend == 'fused':
        return fused_triton(x, lam_re, lam_im, gain, c_re, c_im, reset)
    sr, si = modes_scan(x, lam_re, lam_im, gain, reset, backend='reference' if backend == 'reference' else 'auto')
    return (sr * c_re).sum(-1) - (si * c_im).sum(-1)


def _initial_values(k, cplx):
    """Deterministic per-group initial timescales and angles (no RNG)."""
    taus, thetas = [], []
    for group in range(_GROUPS):
        row_tau, row_theta = [], []
        for mode in range(k):
            position = ((group + mode * _GROUPS // 2) % _GROUPS) / (_GROUPS - 1)
            row_tau.append(2. * 256. ** position)
            row_theta.append((math.pi / 4) * ((group * 7 + mode * 3) % _GROUPS) / (_GROUPS - 1) if cplx else 0.)
        taus.append(row_tau)
        thetas.append(row_theta)
    return taus, thetas


class DynamicsFFN(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        dynamics: str = 'c1'
        placement: str = 'gate'

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import dense_param_placement
                names = ('nu', 'c_re') + (('theta', 'c_im') if MODES[self.dynamics][1] else ())
                self.sharding_config = replace(self.sharding_config, state_shardings={
                    **self.sharding_config.state_shardings,
                    **{n: dense_param_placement(tp=spmd.R) for n in names}})
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        nn.Module.__init__(self)
        if config.dynamics not in MODES or config.placement not in PLACEMENTS:
            raise ValueError((config.dynamics, config.placement))
        self.w1, self.w2, self.w3 = config.w1.build(), config.w2.build(), config.w3.build()
        self.dynamics, self.placement = config.dynamics, config.placement
        k, cplx = MODES[config.dynamics]
        h = config.w1.out_features
        self.nu = nn.Parameter(torch.empty(h, k))
        self.c_re = nn.Parameter(torch.empty(h, k))
        self.theta = nn.Parameter(torch.empty(h, k)) if cplx else None
        self.c_im = nn.Parameter(torch.empty(h, k)) if cplx else None
        self.capture_diagnostics, self.diagnostics = False, {}

    def reset_parameters(self):
        h, k = self.nu.shape
        taus, thetas = _initial_values(k, self.theta is not None)
        with torch.no_grad():
            self.c_re.zero_()
            if self.c_im is not None:
                self.c_im.zero_()
            for group in range(_GROUPS):
                lo, hi = group * h // _GROUPS, (group + 1) * h // _GROUPS
                for mode in range(k):
                    self.nu[lo:hi, mode].fill_(-math.log(taus[group][mode]))
                    if self.theta is not None:
                        self.theta[lo:hi, mode].fill_(thetas[group][mode])

    def modes(self, u, positions):
        with torch.autocast(device_type=u.device.type, enabled=False):
            x = u.float() if u.dtype != torch.float64 else u
            decay = torch.exp(-torch.exp(self.nu.to(x)))
            if os.environ.get('SA_NULL_MEMORY'):  # zero-memory control: lambda = 0, same parameters
                decay = decay * 0.
            if self.theta is not None:
                theta = self.theta.to(x)
                lam_re, lam_im = decay * torch.cos(theta), decay * torch.sin(theta)
            else:
                lam_re, lam_im = decay, torch.zeros_like(decay)
            c_re = self.c_re.to(x)
            c_im = self.c_im.to(x) if self.c_im is not None else torch.zeros_like(c_re)
            out = modes_readout(x, lam_re, lam_im, 1. - decay, c_re, c_im, positions == 0)
            if self.capture_diagnostics:
                self.diagnostics = {
                    'c_norm': torch.linalg.vector_norm(self.c_re.detach().float()) if self.c_im is None
                    else torch.linalg.vector_norm(torch.stack((self.c_re.detach().float(), self.c_im.detach().float()))),
                    'tau_median': torch.median(torch.exp(-self.nu.detach().float())),
                }
            return out

    def forward(self, x, *, positions=None):
        if positions is None:
            raise ValueError('Dynamics FFN requires explicit document positions')
        gate, up = self.w1(x), self.w3(x)
        if self.placement == 'gate':
            y = F.silu(gate + self.modes(gate, positions).to(gate.dtype)) * up
        elif self.placement == 'up':
            y = F.silu(gate) * (up + self.modes(up, positions).to(up.dtype))
        else:
            h = F.silu(gate) * up
            y = h + self.modes(h, positions).to(h.dtype)
        return self.w2(y)


BIOPHYS = {'alif': 2, 'adex': 2, 'hh': 4}  # state variables per neuron
_MICRO = {'alif': 4, 'adex': 4, 'hh': 10}
_CHUNK = 8


class _Spike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1. + 4. * x.abs()) ** 2


def _spike(x):
    return _Spike.apply(x)


def _alif_chunk(v, a, current, keep, p, micro):
    """ALIF (dimensionless): leaky membrane, soft reset at threshold 1, spike-triggered adaptation."""
    alpha, rho, beta = p
    outs_v, outs_a = [], []
    for i in range(current.shape[1]):
        k = keep[:, i, None]
        v, a = v * k, a * k
        for _ in range(micro):
            v = alpha * v + (1. - alpha) * (current[:, i] - beta * a)
            z = _spike(v - 1.)
            v = v - z
            a = rho * a + z
        outs_v.append(v)
        outs_a.append(a)
    return v, a, torch.stack(outs_v, 1), torch.stack(outs_a, 1)


def _adex_chunk(v, w, current, keep, p, micro):
    """AdEx (dimensionless: E_L=0, V_T=1, Delta_T=0.25, V_peak=3, V_reset=0), explicit Euler micro-steps."""
    inv_tm, inv_tw, a_c, b_c = p
    outs_v, outs_w = [], []
    for i in range(current.shape[1]):
        k = keep[:, i, None]
        v, w = v * k, w * k
        for _ in range(micro):
            # Spike initiation: exact exponential in the forward pass; its steep slope is excluded from BPTT
            # (the spike surrogate carries the gradient), otherwise gradients explode through time.
            e = (0.25 * torch.exp((torch.clamp(v, max=3.) - 1.) / 0.25)).detach()
            v = v + inv_tm * (-v + e - w + current[:, i])
            w = w + inv_tw * (a_c * v - w)
            z = _spike(v - 3.)
            v = v - v * z
            w = w + b_c * z
        outs_v.append(v)
        outs_w.append(w)
    return v, w, torch.stack(outs_v, 1), torch.stack(outs_w, 1)


def _hh_rates(v):
    x = v + 65.
    am = 0.1 * (x - 25.) / (1. - torch.exp(-(x - 25.) / 10.) + 1e-6)
    bm = 4. * torch.exp(-x / 18.)
    ah = 0.07 * torch.exp(-x / 20.)
    bh = 1. / (1. + torch.exp(-(x - 30.) / 10.))
    an = 0.01 * (x - 10.) / (1. - torch.exp(-(x - 10.) / 10.) + 1e-6)
    bn = 0.125 * torch.exp(-x / 80.)
    return am, bm, ah, bh, an, bn


def _hh_chunk(state, current, keep, p, micro):
    """Hodgkin-Huxley squid axon (mV, ms, uA/cm^2); one token = 1 ms; exponential-Euler gates."""
    v, m, h, n = state
    bias, gain = p
    dt = 1. / micro
    rest = (-65., 0.0529, 0.5961, 0.3177)
    outs = [[], [], [], []]
    for i in range(current.shape[1]):
        k = keep[:, i, None]
        v, m, h, n = [s * k + r * (1. - k) for s, r in zip((v, m, h, n), rest)]
        drive = bias + gain * current[:, i]
        for _ in range(micro):
            # Gate kinetics see a detached voltage: forward dynamics are exact HH, but BPTT flows through the
            # (positive) membrane conductances only, which keeps gradients bounded through time.
            am, bm, ah, bh, an, bn = _hh_rates(v.detach())
            m = m + (am / (am + bm) - m) * (1. - torch.exp(-(am + bm) * dt))
            h = h + (ah / (ah + bh) - h) * (1. - torch.exp(-(ah + bh) * dt))
            n = n + (an / (an + bn) - n) * (1. - torch.exp(-(an + bn) * dt))
            ion = 120. * m ** 3 * h * (v - 50.) + 36. * n ** 4 * (v + 77.) + 0.3 * (v + 54.387)
            v = torch.clamp(v + dt * (drive - ion), -100., 60.)
        for o, s in zip(outs, (v, m, h, n)):
            o.append(s)
    return (v, m, h, n), [torch.stack(o, 1) for o in outs]


_COMPILED = {}


def _compiled(name, fn):
    if os.environ.get('SA_BIOPHYS_EAGER'):
        return fn
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(fn, dynamic=False, fullgraph=False)
    return _COMPILED[name]


class BiophysFFN(FeedForward):
    """Biophysical membrane (ALIF / AdEx / HH) on the somatic current, gate placement, zero-init readout."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        model: str = 'alif'

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import dense_param_placement
                self.sharding_config = replace(self.sharding_config, state_shardings={
                    **self.sharding_config.state_shardings,
                    **{n: dense_param_placement(tp=spmd.R) for n in ('bp', 'readout')}})
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        nn.Module.__init__(self)
        if config.model not in BIOPHYS:
            raise ValueError(config.model)
        self.w1, self.w2, self.w3 = config.w1.build(), config.w2.build(), config.w3.build()
        self.model = config.model
        h = config.w1.out_features
        self.bp = nn.Parameter(torch.empty(h, 4))  # raw per-neuron biophysical parameters
        self.readout = nn.Parameter(torch.empty(h, BIOPHYS[config.model]))
        self.capture_diagnostics, self.diagnostics = False, {}

    def reset_parameters(self):
        h = self.bp.shape[0]
        with torch.no_grad():
            self.readout.zero_()
            for group in range(_GROUPS):
                lo, hi = group * h // _GROUPS, (group + 1) * h // _GROUPS
                f = group / (_GROUPS - 1)
                if self.model == 'alif':    # log tau_v in tokens, log tau_a, softplus^-1 beta
                    vals = (math.log(2. * 32. ** f), math.log(16. * 32. ** f), math.log(math.expm1(.5)), 0.)
                elif self.model == 'adex':  # log tau_m, log tau_w, softplus^-1 a, softplus^-1 b
                    vals = (math.log(1. + 7. * f), math.log(16. * 16. ** f), math.log(math.expm1(.1)), math.log(math.expm1(.2)))
                else:                       # bias current, log gain
                    vals = (2. + 4. * f, math.log(3.), 0., 0.)
                for j, value in enumerate(vals):
                    self.bp[lo:hi, j].fill_(value)

    def membrane(self, current, positions):
        with torch.autocast(device_type=current.device.type, enabled=False):
            x = current.float()
            b, t, h = x.shape
            keep = (positions != 0).to(x.dtype)
            raw = self.bp.to(x)
            micro = _MICRO[self.model]
            outs = []
            if self.model == 'alif':
                p = (torch.exp(-1. / (micro * torch.exp(raw[:, 0]))), torch.exp(-1. / (micro * torch.exp(raw[:, 1]))),
                     F.softplus(raw[:, 2]))
                s0, s1 = x.new_zeros(b, h), x.new_zeros(b, h)
                fn = _compiled('alif', _alif_chunk)
            elif self.model == 'adex':
                p = (1. / (micro * torch.exp(raw[:, 0])), 1. / (micro * torch.exp(raw[:, 1])),
                     F.softplus(raw[:, 2]), F.softplus(raw[:, 3]))
                s0, s1 = x.new_zeros(b, h), x.new_zeros(b, h)
                fn = _compiled('adex', _adex_chunk)
            else:
                p = (raw[:, 0], torch.exp(raw[:, 1]))
                state = tuple(x.new_full((b, h), r) for r in (-65., 0.0529, 0.5961, 0.3177))
                fn = _compiled('hh', _hh_chunk)
            chunk = 2 if self.model == 'hh' else _CHUNK
            for start in range(0, t, chunk):
                cur, kp = x[:, start:start + chunk], keep[:, start:start + chunk]
                if self.model == 'hh':
                    state, seq = fn(state, cur, kp, p, micro)
                    feats = [(seq[0] + 65.) / 30., seq[1], seq[2], seq[3]]
                else:
                    s0, s1, v_seq, a_seq = fn(s0, s1, cur, kp, p, micro)
                    feats = [v_seq, a_seq]
                outs.append(sum(f * self.readout[:, j].to(x) for j, f in enumerate(feats)))
            return torch.cat(outs, 1)

    def forward(self, x, *, positions=None):
        if positions is None:
            raise ValueError('Biophysical FFN requires explicit document positions')
        gate, up = self.w1(x), self.w3(x)
        return self.w2(F.silu(gate + self.membrane(gate, positions).to(gate.dtype)) * up)


def residual_local_parameters(a):
    return a.dim * (4 * a.residual_modes + 1)


DENDRITE_KINDS = ('full', 'uniform', 'learned', 'linear')


def dendrite_local_parameters(a):
    b = a.dendrite_branches
    gamma = 0 if a.dendrite_kind == 'linear' else a.ff_dim * b
    return gamma + (a.dim * a.dim if a.dendrite_kind == 'learned' else 0)


def _local(t):
    from torch.distributed.tensor import DTensor
    return t.to_local() if isinstance(t, DTensor) else t


class BranchFFN(FeedForward):
    """Dendritic branches on the somatic (gate) current, SwiGLU analog path unchanged.

    full:    every branch sees all inputs (more synapses per neuron, fewer neurons)
    uniform: branch j sees input block j (same synapse count as a point neuron)
    learned: like uniform after a learned per-layer rotation (identity at init)
    linear:  uniform with gamma fixed to zero, i.e. exactly the point neuron
    Branch response u + gamma * silu(u); gamma is zero at initialisation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        kind: str = 'uniform'
        branches: int = 4

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import dense_param_placement
                names = (() if self.kind == 'linear' else ('gamma',)) + (('mix',) if self.kind == 'learned' else ())
                self.sharding_config = replace(self.sharding_config, state_shardings={
                    **self.sharding_config.state_shardings,
                    **{n: dense_param_placement(tp=spmd.R) for n in names}})
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        nn.Module.__init__(self)
        if config.kind not in DENDRITE_KINDS:
            raise ValueError(config.kind)
        self.w1, self.w2, self.w3 = config.w1.build(), config.w2.build(), config.w3.build()
        self.kind, self.branches = config.kind, config.branches
        hidden, dim = config.w3.out_features, config.w3.in_features
        if dim % self.branches:
            raise ValueError('model width must be divisible by the branch count')
        self.gamma = None if self.kind == 'linear' else nn.Parameter(torch.empty(hidden, self.branches))
        self.mix = nn.Parameter(torch.empty(dim, dim)) if self.kind == 'learned' else None
        self.capture_diagnostics, self.diagnostics = False, {}

    def reset_parameters(self):
        with torch.no_grad():
            if self.gamma is not None:
                self.gamma.fill_(1. if self.kind == 'dsq' else 0.)
            if self.mix is not None:
                self.mix.zero_()
                for i in range(self.mix.shape[0]):
                    self.mix[i, i].fill_(1.)

    def branch_currents(self, x, dtype):
        b = self.branches
        if self.kind == 'full':
            u = self.w1(x)
            return u.unflatten(-1, (-1, b))
        z = F.linear(x, _local(self.mix).to(x.dtype)) if self.kind == 'learned' else x
        w = _local(self.w1.weight)
        h, d = w.shape
        # Same compute precision as the baseline nn.Linear (dtype of the analog path).
        zb = z.to(dtype).unflatten(-1, (b, d // b))
        return torch.einsum('btjk,hjk->bthj', zb, w.view(h, b, d // b).to(dtype))

    def forward(self, x, *, positions=None):
        up = self.w3(x)
        u = self.branch_currents(x, up.dtype)
        if os.environ.get('SA_BRANCH_FP32_SUM'):
            u = u.float()
        if self.gamma is None:
            v = u.sum(-1)
        else:
            v = (u + _local(self.gamma).to(u.dtype) * F.silu(u)).sum(-1)
        v = v.to(up.dtype)
        if self.kind == 'full':
            v = v * self.branches ** -.5
        if os.environ.get('SA_DEBUG_DTYPE') and not getattr(self, '_printed', False):
            self._printed = True
            w = _local(self.w1.weight)
            print('DTYPE', dict(x=str(x.dtype), weight=str(w.dtype), up=str(up.dtype), u=str(u.dtype),
                                old_einsum=str(torch.einsum('btk,hk->bth', x, w.to(x.dtype)).dtype),
                                linear=str(self.w1(x).dtype)), flush=True)
        if self.capture_diagnostics and self.gamma is not None:
            self.diagnostics = {'gamma_norm': torch.linalg.vector_norm(self.gamma.detach().float())}
        return self.w2(F.silu(v) * up)


SPARSE_KINDS = ('point', 'dend', 'dlin', 'dplat', 'dsq', 'ndend', 'nsq')


def sparse_fanin(a):
    return round(a.fanin * a.dim)


def sparse_local_parameters(a):
    b = a.dendrite_branches
    return a.ff_dim * {'dend': b, 'dplat': 2 * b, 'dsq': b, 'ndend': 1}.get(a.sparse_kind, 0)


def fanin_synapses(hidden, dim, fanin, branches, seed):
    """Balanced random fan-in. Every neuron reads `fanin` distinct residual channels and every channel is
    read by exactly hidden * fanin / dim neurons: each random permutation of the channels is cut into
    dim / fanin consecutive chunks, one chunk per neuron. A neuron's chunk (in random order) is split into
    `branches` equal branches. Returns int8 [hidden, dim] holding the branch of each synapse, -1 if absent."""
    if dim % fanin or hidden % (dim // fanin) or fanin % branches:
        raise ValueError(f'unbalanced fan-in: hidden={hidden} dim={dim} fanin={fanin} branches={branches}')
    g = torch.Generator().manual_seed(seed)
    chunks = torch.cat([torch.randperm(dim, generator=g) for _ in range(hidden // (dim // fanin))])
    channels = chunks.view(hidden, fanin)
    branch = (torch.arange(fanin) // (fanin // branches)).to(torch.int8).expand(hidden, fanin)
    out = torch.full((hidden, dim), -1, dtype=torch.int8)
    out.scatter_(1, channels, branch)
    return out


class SparseFFN(FeedForward):
    """SwiGLU neurons with limited fan-in (the spatial communication budget).

    Each neuron reads `fanin` of the `dim` residual channels; the gate w1 and the analog path w3 share
    these synapses. Weights outside them are zero and stay zero (masked in the forward pass).
    point: gate current u = sum over the neuron's synapses.
    dend:  the synapses form `branches` equal branches; gate v = sum_b u_b + gamma_b * silu(u_b),
           gamma zero at initialisation (the Stage A uniform dendrite on a sparse neuron).
    dlin:  the dend code path with gamma fixed to zero, i.e. exactly the point neuron.
    dplat: plateau branches (NMDA-like, thresholded and saturating): v = sum_b u_b + gamma_b * sigmoid(4 (u_b - theta_b)),
           gamma zero and theta 0.5 at initialisation (the stronger dendrite of the pre-registered follow-up).
    dsq:   supralinear branches (Poirazi-style subunits): v = sum_b u_b + gamma_b * relu(u_b)^2, gamma one at
           initialisation.
    Non-gated neurons (no SwiGLU product):
    ndend: Poirazi-Mel two-layer neuron. Both w1 and w3 are dendritic weights: on each of the `branches` synapse
           groups the neuron has two branch projections (from w1 and from w3), 2 * branches subunits in all, each
           relu(u)^2; the soma thresholds their mean, h = relu(mean relu(u)^2 - theta). theta (one per neuron) starts
           at the expected mean at initialisation (mean over branches of |w_branch|^2 / 2 for unit-variance inputs).
           Same synapses and projection parameters as the SwiGLU neuron of the same width.
    nsq:   non-gated point neuron with squared ReLU (Primer), h = relu(u)^2. w3 is not built; equal parameters to
           SwiGLU need 1.5x the width (SA_FF).
    wiring='learned': the synapse placement is learned. Each neuron keeps `fanin / branches` synapses per branch
           (gate path) and `fanin` synapses on the analog path, chosen as the top-k of learned scores (edge-popup
           style straight-through estimator). Scores start at the balanced random map, so a learned run starts from
           exactly the fixed-wiring network. The synapse budget, and the parameter count, are unchanged.
    Kept weights are scaled by sqrt(dim / fanin) at initialisation, so pre-activation variance matches
    the dense layer. The synapse map is a non-persistent buffer rebuilt deterministically from mask_seed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        kind: str = 'point'
        fanin: int = 0
        branches: int = 4
        mask_seed: int = 0
        wiring: str = 'fixed'

        def build(self, **kwargs):
            if self.sharding_config is not None:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import dense_param_placement
                names = {'dend': ('synapse', 'gamma'), 'dplat': ('synapse', 'gamma', 'theta'),
                         'dsq': ('synapse', 'gamma'), 'ndend': ('synapse', 'theta')}.get(self.kind, ('synapse',))
                if self.wiring == 'learned':
                    names = names + ('score1', 'score3')
                self.sharding_config = replace(self.sharding_config, state_shardings={
                    **self.sharding_config.state_shardings, **{n: dense_param_placement(tp=spmd.R) for n in names}})
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        nn.Module.__init__(self)
        if (config.kind not in SPARSE_KINDS or config.wiring not in ('fixed', 'learned')
                or (config.kind in ('ndend', 'nsq') and config.wiring != 'fixed')):
            raise ValueError((config.kind, config.wiring))
        self.w1, self.w2 = config.w1.build(), config.w2.build()
        self.w3 = None if config.kind == 'nsq' else config.w3.build()
        hidden, dim = config.w1.out_features, config.w1.in_features
        self.kind, self.fanin, self.mask_seed = config.kind, config.fanin, config.mask_seed
        self.branches = 1 if config.kind in ('point', 'nsq') else config.branches
        self.wiring = config.wiring
        self.gamma = nn.Parameter(torch.empty(hidden, self.branches)) if self.kind in ('dend', 'dplat', 'dsq') else None
        self.score1 = nn.Parameter(torch.empty(self.branches, hidden, dim)) if self.wiring == 'learned' else None
        self.score3 = nn.Parameter(torch.empty(hidden, dim)) if self.wiring == 'learned' else None
        self.theta = (nn.Parameter(torch.empty(hidden, self.branches)) if self.kind == 'dplat'
                      else nn.Parameter(torch.empty(hidden)) if self.kind == 'ndend' else None)
        # Reserved empty slot, filled by init_states: a plain (never distributed) tensor rebuilt from the seed.
        self.synapse_shape = (hidden, dim)
        self.register_buffer('synapse', None, persistent=False)
        self.capture_diagnostics, self.diagnostics = False, {}

    def reset_parameters(self):
        with torch.no_grad():
            if self.gamma is not None:
                self.gamma.fill_(1. if self.kind == 'dsq' else 0.)
            if self.theta is not None:
                self.theta.fill_(.5 if self.kind == 'dplat' else 0.)  # ndend: set from the weights in _init_self_buffers

    def build_synapses(self, device):
        hidden, dim = self.synapse_shape
        self.synapse = fanin_synapses(hidden, dim, self.fanin, self.branches, self.mask_seed).to(device)

    def _init_self_buffers(self, *, buffer_device=None):
        # Runs after w1/w3 are initialised: zero the absent synapses, rescale the kept ones.
        device = buffer_device if buffer_device is not None else _local(self.w1.weight).device
        self.build_synapses(device)
        hidden, dim = self.synapse_shape
        with torch.no_grad():
            for w in [m.weight for m in (self.w1, self.w3) if m is not None]:
                local = _local(w)
                if local.shape != self.synapse.shape:
                    raise RuntimeError('SparseFFN needs unsharded w1/w3 (single data-parallel rank)')
                if self.wiring == 'learned':  # every potential synapse keeps a weight; the mask decides
                    local.mul_(math.sqrt(dim / self.fanin))
                else:
                    local.mul_((self.synapse >= 0).to(local.dtype) * math.sqrt(dim / self.fanin))
            if self.kind == 'ndend':  # soma threshold at the expected subunit mean for unit-variance inputs
                w1, w3 = _local(self.w1.weight).float(), _local(self.w3.weight).float()
                sq = [((w * (self.synapse == j)).square().sum(-1) / 2) for w in (w1, w3) for j in range(self.branches)]
                _local(self.theta).copy_(torch.stack(sq).mean(0))
            if self.wiring == 'learned':  # scores U(0,.5) + .5 on the random map: starts at the fixed map, swaps at once
                g = torch.Generator().manual_seed(self.mask_seed + 7)
                noise = lambda shape: (torch.rand(shape, generator=g) * .5).to(self.synapse.device)
                s1, s3 = _local(self.score1), _local(self.score3)
                for j in range(self.branches):
                    s1[j].copy_(noise(s1[j].shape) + .5 * (self.synapse == j).to(s1.dtype))
                s3.copy_(noise(s3.shape) + .5 * (self.synapse >= 0).to(s3.dtype))

    @staticmethod
    def _topk(score, k):
        idx = score.topk(k, dim=-1).indices
        hard = torch.zeros_like(score).scatter_(-1, idx, 1.)
        return hard + score - score.detach()

    def forward(self, x, *, positions=None):
        w1 = _local(self.w1.weight)
        if self.kind == 'nsq':
            return self.w2(F.relu(F.linear(x, w1 * (self.synapse >= 0).to(w1.dtype))).square())
        w3 = _local(self.w3.weight)
        if self.kind == 'ndend':
            masks = [(self.synapse == j).to(w1.dtype) for j in range(self.branches)]
            u = F.linear(x, torch.cat([w * m for w in (w1, w3) for m in masks])).unflatten(-1, (2 * self.branches, -1))
            h = F.relu(F.relu(u).square().mean(-2) - _local(self.theta).to(u.dtype))
            return self.w2(h)
        if self.wiring == 'learned':
            keep = self._topk(_local(self.score3), self.fanin).to(w3.dtype)
            k1 = self.fanin // self.branches
            branch_masks = [self._topk(_local(self.score1)[j], k1).to(w1.dtype) for j in range(self.branches)]
        else:
            keep = (self.synapse >= 0).to(w3.dtype)
            branch_masks = [(self.synapse == j).to(w1.dtype) for j in range(self.branches)]
        up = F.linear(x, w3 * keep)
        if self.kind == 'point':
            gate = F.linear(x, w1 * (branch_masks[0] if self.wiring == 'learned' else keep))
        else:
            stacked = torch.cat([w1 * m for m in branch_masks])
            u = F.linear(x, stacked).unflatten(-1, (self.branches, -1)).transpose(-1, -2)
            if self.kind == 'dplat':
                u = u + _local(self.gamma).to(u.dtype) * torch.sigmoid(4 * (u - _local(self.theta).to(u.dtype)))
            elif self.kind == 'dsq':
                u = u + _local(self.gamma).to(u.dtype) * F.relu(u).square()
            elif self.gamma is not None:
                u = u + _local(self.gamma).to(u.dtype) * F.silu(u)
            gate = u.sum(-1)
        if self.capture_diagnostics and self.gamma is not None:
            self.diagnostics = {'gamma_norm': torch.linalg.vector_norm(self.gamma.detach().float())}
        return self.w2(F.silu(gate) * up)


class ResidualModesBlock(TemporalBlock):
    """Same state budget as neuron-internal modes, but on residual-stream channels before the FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(TemporalBlock.Config):
        residual_modes: int = 6
        rm_norm: object = None

    def __init__(self, config):
        super().__init__(config)
        self.rm_norm = config.rm_norm.build()
        d, k = config.rm_norm.normalized_shape, config.residual_modes
        self.rm_nu = nn.Parameter(torch.empty(d, k))
        self.rm_theta = nn.Parameter(torch.empty(d, k))
        self.rm_c_re = nn.Parameter(torch.empty(d, k))
        self.rm_c_im = nn.Parameter(torch.empty(d, k))

    def reset_parameters(self):
        h, k = self.rm_nu.shape
        taus, thetas = _initial_values(2, True)
        with torch.no_grad():
            self.rm_c_re.zero_()
            self.rm_c_im.zero_()
            for group in range(_GROUPS):
                lo, hi = group * h // _GROUPS, (group + 1) * h // _GROUPS
                for mode in range(k):
                    tau = 2. * 256. ** (((group + mode * _GROUPS // k) % _GROUPS) / (_GROUPS - 1))
                    self.rm_nu[lo:hi, mode].fill_(-math.log(tau))
                    self.rm_theta[lo:hi, mode].fill_(thetas[group][mode % 2])

    def forward(self, x, attention_masks, positions=None):
        if positions is None:
            raise ValueError('Residual modes require explicit document positions')
        h = x + self.attention(self.attention_norm(x), attention_masks, positions)
        u = self.rm_norm(h)
        with torch.autocast(device_type=u.device.type, enabled=False):
            z = u.float()
            decay = torch.exp(-torch.exp(self.rm_nu.to(z)))
            theta = self.rm_theta.to(z)
            extra = modes_readout(z, decay * torch.cos(theta), decay * torch.sin(theta), 1. - decay,
                                  self.rm_c_re.to(z), self.rm_c_im.to(z), positions == 0, backend='fused')
        h = h + extra.to(h.dtype)
        return h + self.feed_forward(self.ffn_norm(h))


def extend_config_stagea(config, a):
    from .neuron_complexity import TemporalBlock, WindowModel
    if a.attention_window not in WINDOWS:
        raise ValueError(f'Stage A window must be one of {WINDOWS}')
    if a.dynamics:
        if (a.dynamics not in MODES or a.placement not in PLACEMENTS or a.activation != 'swiglu'
                or a.ffn_kind != 'dense' or a.temporal or a.r4_scales or a.attention_kind != 'softmax'):
            raise ValueError('Invalid Stage A dynamics architecture')
        for i, layer in enumerate(config.layers):
            f = layer.feed_forward
            layer.feed_forward = DynamicsFFN.Config(w1=f.w1, w2=f.w2, w3=f.w3,
                                                    dynamics=a.dynamics, placement=a.placement)
            config.layers[i] = TemporalBlock.Config(**{v.name: getattr(layer, v.name) for v in fields(layer)})
    elif a.depth_groups:
        if (a.attention_window or a.activation != 'swiglu' or a.ffn_kind != 'dense' or a.temporal or a.r4_scales
                or a.attention_kind != 'softmax' or (a.dim * a.residual_mult) % a.depth_groups):
            raise ValueError('Invalid depth-dendrite architecture')
        from torchtitan.models.common import RMSNorm
        from torchtitan.models.llama3 import _NORM_INIT
        from .depth_dendrites import DepthBlock, DepthModel
        for i, layer in enumerate(config.layers):
            config.layers[i] = DepthBlock.Config(
                **{v.name: getattr(layer, v.name) for v in fields(layer)}, layer_id=i, n_layers=a.layers,
                groups=a.depth_groups, dim=a.dim, input_norm=RMSNorm.Config(normalized_shape=a.dim, param_init=_NORM_INIT),
                mode=a.depth_mode, hidden_mult=a.depth_hidden_mult, residual=a.dim * a.residual_mult,
                lns=a.depth_lns, mix_norm=a.depth_mix_norm)
        config = DepthModel.Config(**{v.name: getattr(config, v.name) for v in fields(config)},
                                   depth_groups=a.depth_groups, residual=a.dim * a.residual_mult)
    elif a.hc_streams:
        if (a.attention_window or a.activation != 'swiglu' or a.ffn_kind != 'dense'
                or a.temporal or a.r4_scales or a.attention_kind != 'softmax'
                or a.residual_mult != 1 or a.depth_lns):
            raise ValueError('Invalid HC architecture')
        from .hyper_connections import HCBlock, HCModel
        for i, layer in enumerate(config.layers):
            config.layers[i] = HCBlock.Config(
                **{v.name: getattr(layer, v.name) for v in fields(layer)},
                layer_id=i, dim=a.dim, streams=a.hc_streams)
        config = HCModel.Config(**{v.name: getattr(config, v.name) for v in fields(config)},
                                streams=a.hc_streams)
    elif a.attnres:
        if (a.attention_window or a.activation != 'swiglu' or a.ffn_kind != 'dense' or a.temporal or a.r4_scales
                or a.attention_kind != 'softmax' or a.residual_mult != 1 or a.depth_lns):
            raise ValueError('Invalid attention-residual architecture')
        from .attn_residuals import AttnResBlock, AttnResModel
        if a.attnres_block_size:
            from .block_attn_residuals import BlockAttnResBlock as AttnResBlock, BlockAttnResModel as AttnResModel
        for i, layer in enumerate(config.layers):
            config.layers[i] = AttnResBlock.Config(
                **{v.name: getattr(layer, v.name) for v in fields(layer)}, layer_id=i, n_layers=a.layers, dim=a.dim,
                heads=a.attnres_heads)
        config = AttnResModel.Config(**{v.name: getattr(config, v.name) for v in fields(config)},
            **({'attnres_block_size': a.attnres_block_size} if a.attnres_block_size else {}))
    elif a.dca:
        if (a.attention_window or a.activation != 'swiglu' or a.ffn_kind != 'dense' or a.temporal or a.r4_scales
                or a.attention_kind != 'softmax' or a.residual_mult != 1 or a.depth_lns):
            raise ValueError('Invalid DeepCrossAttention architecture')
        from torchtitan.models.common import RMSNorm
        from torchtitan.models.llama3 import _NORM_INIT
        from .dca import DCABlock, DCAModel
        for i, layer in enumerate(config.layers):
            config.layers[i] = DCABlock.Config(
                **{v.name: getattr(layer, v.name) for v in fields(layer)}, layer_id=i, n_layers=a.layers, dim=a.dim,
                input_norm=RMSNorm.Config(normalized_shape=a.dim, param_init=_NORM_INIT))
        config = DCAModel.Config(**{v.name: getattr(config, v.name) for v in fields(config)})
    elif a.biophys:
        if a.biophys not in BIOPHYS or a.activation != 'swiglu' or a.ffn_kind != 'dense':
            raise ValueError('Invalid Stage B biophysical architecture')
        for i, layer in enumerate(config.layers):
            f = layer.feed_forward
            layer.feed_forward = BiophysFFN.Config(w1=f.w1, w2=f.w2, w3=f.w3, model=a.biophys)
            config.layers[i] = TemporalBlock.Config(**{v.name: getattr(layer, v.name) for v in fields(layer)})
    elif a.residual_modes:
        from torchtitan.models.common import RMSNorm
        from torchtitan.models.llama3 import _NORM_INIT
        block = ResidualModesBlock
        for i, layer in enumerate(config.layers):
            config.layers[i] = block.Config(**{v.name: getattr(layer, v.name) for v in fields(layer)},
                                            residual_modes=a.residual_modes,
                                            rm_norm=RMSNorm.Config(normalized_shape=a.dim, param_init=_NORM_INIT))
    elif a.dendrite_kind:
        if (a.dendrite_kind not in DENDRITE_KINDS or a.activation != 'swiglu' or a.ffn_kind != 'dense'
                or a.temporal or a.r4_scales or a.dim % a.dendrite_branches):
            raise ValueError('Invalid Stage A dendrite architecture')
        for layer in config.layers:
            f = layer.feed_forward
            if a.dendrite_kind == 'full':
                f.w1.out_features = a.ff_dim * a.dendrite_branches
            layer.feed_forward = BranchFFN.Config(w1=f.w1, w2=f.w2, w3=f.w3, kind=a.dendrite_kind,
                                                  branches=a.dendrite_branches)
    elif a.sparse_kind:
        if (a.sparse_kind not in SPARSE_KINDS or a.activation != 'swiglu' or a.ffn_kind != 'dense'
                or a.temporal or a.r4_scales or not 0 < a.fanin <= 1):
            raise ValueError('Invalid spatial-axis architecture')
        for i, layer in enumerate(config.layers):
            f = layer.feed_forward
            layer.feed_forward = SparseFFN.Config(w1=f.w1, w2=f.w2, w3=f.w3, kind=a.sparse_kind,
                                                  fanin=sparse_fanin(a), branches=a.dendrite_branches,
                                                  mask_seed=a.mask_seed * 1000 + i, wiring=a.wiring)
    elif a.temporal or a.r4_scales or a.activation != 'swiglu':
        raise ValueError('Legacy temporal variants use the Step 1/2 path')
    if a.attention_window:
        config = WindowModel.Config(**{v.name: getattr(config, v.name) for v in fields(config)},
                                    window=a.attention_window)
    return config
