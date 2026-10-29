"""Static Huber proximal screening. No state crosses tokens.

Rows store local vectors; D is shared within each layer. Ref uses 32 contractive
steps (eta=1/2, lambda<=1, ||D||<=1) and implicit first derivatives. Its residual
is checked independently, not assumed zero. Huber boundary derivative uses the
inside convention, matching torch.clamp. No dense projection is recomputed.
"""
import torch
import torch.nn.functional as F

E_KINDS = ("e_quadratic", "e_closed", "e_k1", "e_k4", "e_ref")
MODELS = ("dense", "mlp", "quadratic", "closed", "k1", "k4", "ref")
REF_STEPS = 32


def orthogonal(raw):
    s = (raw - raw.T) * 0.5
    eye = torch.eye(raw.shape[0], device=raw.device, dtype=raw.dtype)
    return torch.linalg.solve(eye + s, eye - s)


def multiply(x, d, transpose=False):
    b = x.shape[-1]
    return torch.stack(tuple(sum((d[j, i] if transpose else d[i, j]) * x[..., j]
                                 for j in range(b)) for i in range(b)), -1)


def iterate(u, d, strength, steps):
    q = u
    for _ in range(steps):
        q = q + 0.5 * (u - q - strength * multiply(multiply(q, d).clamp(-1, 1), d, True))
    return q


def solve_spd(rhs, d, mask, strength):
    """Coordinate Cholesky; mask may be token dependent. No explicit inverse."""
    b = rhs.shape[-1]
    lower = {}
    for i in range(b):
        for j in range(i + 1):
            v = float(i == j) + strength * sum(d[k, i] * mask[..., k] * d[k, j] for k in range(b))
            for k in range(j):
                v = v - lower[i, k] * lower[j, k]
            lower[i, j] = v.sqrt() if i == j else v / lower[j, j]
    z, y = [], {}
    for i in range(b):
        v = rhs[..., i]
        for j in range(i):
            v = v - lower[i, j] * z[j]
        z.append(v / lower[i, i])
    for i in reversed(range(b)):
        v = z[i]
        for j in range(i + 1, b):
            v = v - lower[j, i] * y[j]
        y[i] = v / lower[i, i]
    return torch.stack(tuple(y[i] for i in range(b)), -1)


class ImplicitProx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, d, strength):
        q = iterate(u, d, strength, REF_STEPS)
        ctx.save_for_backward(q, d)
        ctx.strength = strength
        return q

    @staticmethod
    def backward(ctx, grad):
        q, d = ctx.saved_tensors
        z = multiply(q, d)
        m = (z.abs() <= 1).to(q.dtype)
        v = solve_spd(grad, d, m, ctx.strength)
        dv = multiply(v, d)
        axes = tuple(range(q.ndim - 1))
        dd = torch.stack(tuple(torch.stack(tuple(
            (-ctx.strength * (z[..., i].clamp(-1, 1) * v[..., j]
                              + m[..., i] * dv[..., i] * q[..., j])).sum(axes)
            for j in range(d.shape[1]))) for i in range(d.shape[0])))
        return v, dd, None


def apply(u, d, kind, strength=1.0):
    if kind == "e_closed":
        z = multiply(u, d)
        y = torch.where(z.abs() <= 1 + strength, z / (1 + strength), z - strength * z.sign())
        return multiply(y, d, True)
    if kind == "e_quadratic":
        # The factor is shared by all groups, so its construction is amortized.
        return solve_spd(u, d, torch.ones_like(d[0]), strength)
    if kind == "e_ref":
        return ImplicitProx.apply(u, d, strength)
    if kind in ("e_k1", "e_k2", "e_k4", "e_k8"):
        return iterate(u, d, strength, int(kind[3:]))
    raise ValueError(kind)


def oracle(u, d, kind, strength=1.0):
    """Independent library-matmul FP64 reference; autograd through convergence."""
    if kind == "e_closed":
        z = F.linear(u, d)
        return F.linear(torch.where(z.abs() <= 1 + strength, z / (1 + strength), z - strength*z.sign()), d.T)
    if kind == "e_quadratic":
        a = torch.eye(d.shape[0], dtype=d.dtype, device=d.device) + strength * d.T @ d
        return torch.linalg.solve(a, u.unsqueeze(-1)).squeeze(-1)
    q = u
    for _ in range(128 if kind == "e_ref" else int(kind[3:])):
        q = q + 0.5 * (u - q - strength * F.linear(F.linear(q, d).clamp(-1, 1), d.T))
    return q


_compiled = {}


def compiled(u, d, kind, strength=1.0, dynamic=False):
    if dynamic not in _compiled:
        _compiled[dynamic] = torch.compile(apply, fullgraph=True, dynamic=dynamic)
    return _compiled[dynamic](u, d, kind, strength)


def forward_operations(kind, b=4):
    # Nominal scalar estimate; constraint, factorization/shared work excluded.
    if kind == "e_quadratic":
        return 2*b*b
    if kind == "e_closed":
        return 4*b*b + 4*b
    steps = REF_STEPS if kind == "e_ref" else int(kind[3:])
    return steps * (4*b*b + 5*b)
