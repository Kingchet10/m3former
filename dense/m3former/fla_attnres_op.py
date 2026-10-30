"""Opaque torch.library boundary for the unchanged pinned FLA Triton kernels.

FLA's pointer alignment checks cannot be traced by our fullgraph compiler.
Keep forward/backward kernels intact and provide fake tensors and autograd
registration so the surrounding Transformer remains fully compiled.
"""
import torch
from fla.ops.attnres.fused import _build_ptr_table, fused_attnres_fwd, fused_attnres_bwd


@torch.library.custom_op('rr_fla::forward', mutates_args=())
def routing_forward(q: torch.Tensor, g: torch.Tensor, values: list[torch.Tensor]
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = [v.reshape(-1, v.shape[-1]).contiguous() for v in values]
    o, pre, rstd, logits, lse = fused_attnres_fwd(
        q, flat, _build_ptr_table(flat), g, None, 1e-6, 1.0, 0)
    return o.view_as(values[0]), pre, rstd, logits, lse


@routing_forward.register_fake
def _(q, g, values):
    shape = values[0].shape
    n, d = values[0].numel() // shape[-1], shape[-1]
    return (torch.empty_like(values[0]), values[0].new_empty((n, d)),
            values[0].new_empty((len(values), n), dtype=torch.float32),
            values[0].new_empty((len(values), n), dtype=torch.float32),
            values[0].new_empty((n,), dtype=torch.float32))


@torch.library.custom_op('rr_fla::backward', mutates_args=())
def routing_backward(do: torch.Tensor, q: torch.Tensor, g: torch.Tensor,
                     values: list[torch.Tensor], pre: torch.Tensor,
                     rstd: torch.Tensor, logits: torch.Tensor, lse: torch.Tensor
                     ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    flat = [v.reshape(-1, v.shape[-1]).contiguous() for v in values]
    dvs, dq, dg, _ = fused_attnres_bwd(
        do.reshape_as(flat[0]).contiguous(), q, flat, _build_ptr_table(flat),
        g, None, pre, rstd, logits, lse, 1e-6, 1.0, 0)
    return [dv.view_as(v) for dv, v in zip(dvs, values)], dq, dg


@routing_backward.register_fake
def _(do, q, g, values, pre, rstd, logits, lse):
    return [torch.empty_like(v) for v in values], torch.empty_like(q), torch.empty_like(g)


def setup_context(ctx, inputs, output):
    q, g, values = inputs
    _, pre, rstd, logits, lse = output
    ctx.save_for_backward(q, g, pre, rstd, logits, lse, *values)
    ctx.mark_non_differentiable(pre, rstd, logits, lse)


def backward(ctx, do, *unused):
    q, g, pre, rstd, logits, lse, *values = ctx.saved_tensors
    dvs, dq, dg = routing_backward(do, q, g, values, pre, rstd, logits, lse)
    return dq, dg, dvs


routing_forward.register_autograd(backward, setup_context=setup_context)


def mix(query, residuals, rms_weight):
    return routing_forward(query, rms_weight, list(residuals))[0]
