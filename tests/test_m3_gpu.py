"""Checks of M3 on one GPU: exact initialization, parameter count, fused mixing kernels.

    PYTHONPATH=dense:third_party/fla-kernels python tests/test_m3_gpu.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dense'), str(ROOT / 'third_party/fla-kernels')]

import torch  # noqa: E402

from m3former import m3  # noqa: E402
from m3former.models import Architecture, model_config  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
SMALL = dict(layers=4, dim=256, heads=4, kv_heads=4, ff_dim=704, vocab_size=32000,
             initialization_seed=47, initialization_reference_width=704)


def build(**kw):
    with torch.device('cuda'):
        model = model_config(Architecture(**SMALL, **kw)).build()
    model.init_states(buffer_device=torch.device('cuda'))
    return model


def logits(model, x):
    pos = torch.arange(x.shape[1], device=x.device)[None].expand_as(x).contiguous()
    with torch.no_grad():
        return model(x, positions=pos, attention_masks=model.get_attention_masks(positions=pos)).float()


def test_exact_initialization():
    """At initialization, M3Former, MUDDFormer and multi-group mixing compute the Transformer."""
    x = torch.randint(0, 32000, (2, 256), device='cuda')
    base = build()
    reference = logits(base, x)
    for groups, mode in [(1, 'dense'), (16, 'dense'), (64, 'dense'), (16, 'dense1')]:
        model = build(depth_groups=groups, depth_mode=mode)
        shared = base.state_dict()
        for name, tensor in model.state_dict().items():
            if name in shared:  # parameter-name seeding gives the same backbone; check it
                assert torch.equal(tensor, shared[name]), name
        # Separate Q/K/V projections from the fused weight can round differently from one fused GEMM.
        torch.testing.assert_close(logits(model, x), reference, atol=1e-4, rtol=1e-4)
        print(f'G={groups} {mode}: matches the Transformer at initialization')


def test_parameter_count():
    base = sum(p.numel() for p in build().parameters())
    for groups in (1, 4, 16, 64):
        for mode in m3.MODES:
            model = build(depth_groups=groups, depth_mode=mode)
            added = sum(p.numel() for p in model.parameters()) - base
            assert added == m3.m3_parameters(SMALL['dim'], SMALL['layers'], groups, mode), (groups, mode)
    print('parameter counts match m3.m3_parameters')


def test_fused_mixing():
    """The Triton kernel computes the same mixture and gradients as the PyTorch loop."""
    from m3former.mix_kernel import mix

    torch.manual_seed(0)
    b, t, d = 2, 128, 256
    for streams, sources, groups in [(4, 5, 16), (4, 3, 1), (1, 6, 64)]:
        hs = [torch.randn(b, t, d, device='cuda', requires_grad=True) for _ in range(sources)]
        w = torch.randn(b, t, streams, sources, groups, device='cuda', requires_grad=True)
        out = mix(hs, w, streams, groups)
        ref = torch.stack([sum(w[:, :, c, j, :, None] * h.view(b, t, groups, d // groups)
                               for j, h in enumerate(hs)).reshape(b, t, d) for c in range(streams)])
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        g = torch.randn_like(out)
        fused = torch.autograd.grad(out, [w, *hs], g)
        loop = torch.autograd.grad(ref, [w, *hs], g)
        for u, v in zip(fused, loop):
            torch.testing.assert_close(u, v, atol=1e-4, rtol=1e-4)
    print('fused kernel matches the PyTorch loop (forward and backward)')


def test_v2_kernels():
    """mix_kernel_v2 (group-spanning tiles, in-place accumulation of source gradients) against the PyTorch loop:
    loss and every parameter gradient of a trained-looking model, eager and with per-block torch.compile
    (fullgraph, as in training)."""
    torch.manual_seed(0)
    x = torch.randint(0, 32000, (2, 256), device='cuda')
    y = torch.randint(0, 32000, (2, 256), device='cuda')
    pos = torch.arange(x.shape[1], device='cuda')[None].expand_as(x).contiguous()

    def run(model, kernels, fused=True):
        m3.KERNELS, m3.FUSED = kernels, fused
        model.zero_grad(set_to_none=True)
        out = model(x, positions=pos, attention_masks=model.get_attention_masks(positions=pos))
        loss = torch.nn.functional.cross_entropy(out.float().flatten(0, 1), y.flatten())
        loss.backward()
        return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    for groups, mode in [(1, 'dense'), (16, 'dense'), (64, 'dense'), (16, 'dense1'), (16, 'static')]:
        model = build(depth_groups=groups, depth_mode=mode)
        with torch.no_grad():  # move away from the exact residual connection
            for name, p in model.named_parameters():
                if name.endswith(('_w2', '_a')):
                    p.add_(torch.randn_like(p) * 0.2)
        ref_loss, ref = run(model, 'v1', fused=False)
        for label, kernels in [('v1', 'v1'), ('v2', 'v2')]:
            loss, grads = run(model, kernels)
            torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-5)
            assert grads.keys() == ref.keys()
            for n in ref:  # FP32; the kernels sum in a different order than the loop (max |grad| ~ 2)
                torch.testing.assert_close(grads[n], ref[n], atol=1e-4, rtol=1e-4, msg=f'{label} {n}')
        for layer in model.layers.values():
            layer.compile(fullgraph=True)
        loss, grads = run(model, 'v2')
        torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)
        for n in ref:
            torch.testing.assert_close(grads[n], ref[n], atol=1e-4, rtol=1e-3, msg=f'compiled v2 {n}')
        print(f'G={groups} {mode}: v1 and v2 kernels (eager and compiled) match the PyTorch loop')
    m3.KERNELS, m3.FUSED = 'v2', True


if __name__ == '__main__':
    test_exact_initialization()
    test_parameter_count()
    test_fused_mixing()
    test_v2_kernels()
    print('ok')
