"""Input-dependent SPD local operators, without compression.

Shape suffixes: N = any leading token/group dimensions, B = group width,
J = MLP bottleneck. Vectors are stored as rows; D acts as D @ u in the
column-vector definition. Gates always use the original input coordinates.
"""

import torch
import torch.nn.functional as F

P_KINDS = ("p_scale", "p_diag", "p_direct", "p_k1", "p_k4", "p_mlp")
TRAIN_MODELS = ("p_dense", "p_mlp", "p_scale", "p_diag", "p_direct", "p_k1", "p_k4")


def gate(u_NB):
    # hypot avoids overflow and preserves tiny gates without subtracting from 1.
    return (u_NB / torch.hypot(u_NB, torch.ones_like(u_NB))).square()


def constrain(raw_BB):
    """Exact spectral norm clipping, including its autograd derivative."""
    norm = (
        raw_BB.abs().amax()
        if raw_BB.ndim == 1
        else torch.linalg.matrix_norm(raw_BB, ord=2)
    )
    return raw_BB / norm.clamp_min(1.0)


def initial_d(width, *, device=None, dtype=torch.float32):
    """Nonzero, nonorthogonal, distinct spectrum, strictly inside the norm ball."""
    return torch.tensor(initial_d_values(width), device=device, dtype=dtype)


def initial_d_values(width):
    """Scalar initializer values also support already distributed parameters."""
    return tuple(
        tuple(
            (0.35 + 0.35 * i / (width - 1) if i == j else 0.0)
            + 0.025 * ((i + 2 * j) % 5 - 2) / width
            for j in range(width)
        )
        for i in range(width)
    )


def explicit_system(u_NB, d_BB, alpha=1.0):
    a_NB = gate(u_NB)
    return torch.eye(u_NB.shape[-1], device=u_NB.device, dtype=u_NB.dtype) + alpha * (
        d_BB.T @ (a_NB.unsqueeze(-1) * d_BB)
    )


def apply_reference(u_NB, d_BB, kind, alpha=1.0, w1_JB=None, w2_BJ=None):
    """Batched library reference; d is already norm-constrained or orthogonal."""
    if kind in ("p_dense", "p_i"):
        return u_NB
    if kind == "p_mlp":
        return u_NB + F.linear(F.silu(F.linear(u_NB, w1_JB)), w2_BJ)
    a_NB = gate(u_NB)
    if kind == "p_scale":
        return u_NB / (1.0 + alpha * a_NB.mean(-1, keepdim=True))
    if kind == "p_diag":
        return u_NB / (1.0 + alpha * d_BB.square() * a_NB)
    if kind == "p_ortho":
        return F.linear(F.linear(u_NB, d_BB) / (1.0 + alpha * a_NB), d_BB.T)
    if kind == "p_direct":
        system_NBB = explicit_system(u_NB, d_BB, alpha)
        factor_NBB = torch.linalg.cholesky(system_NBB)
        intermediate_NB1 = torch.linalg.solve_triangular(
            factor_NBB, u_NB.unsqueeze(-1), upper=False
        )
        return torch.linalg.solve_triangular(
            factor_NBB.transpose(-1, -2), intermediate_NB1, upper=True
        ).squeeze(-1)
    if kind not in ("p_k1", "p_k2", "p_k4", "p_k8"):
        raise ValueError("Unknown local operator")
    q_NB = u_NB
    for _ in range(int(kind[3:])):
        q_NB = q_NB + 0.5 * (
            u_NB - q_NB - alpha * F.linear(a_NB * F.linear(q_NB, d_BB), d_BB.T)
        )
    return q_NB


def apply_elementwise(u_NB, d_BB, kind, alpha=1.0, w1_JB=None, w2_BJ=None):
    """Same equations, unrolled only over B coordinates, never over groups.

    Cholesky plus two triangular substitutions computes the Direct solution.
    Autograd differentiates this program, including the input-dependent gate.
    Inductor may fuse local arithmetic; no custom backward or explicit inverse.
    """
    width = u_NB.shape[-1]
    u = tuple(u_NB[..., i] for i in range(width))
    if kind == "p_mlp":
        hidden = tuple(
            F.silu(sum(w1_JB[j, i] * u[i] for i in range(width)))
            for j in range(w1_JB.shape[0])
        )
        return torch.stack(
            tuple(
                u[i] + sum(w2_BJ[i, j] * hidden[j] for j in range(len(hidden)))
                for i in range(width)
            ),
            -1,
        )
    if kind in ("p_dense", "p_i", "p_scale", "p_diag"):
        return apply_reference(u_NB, d_BB, kind, alpha, w1_JB, w2_BJ)
    a_NB = gate(u_NB)
    a = tuple(a_NB[..., i] for i in range(width))

    def multiply(q, transpose=False):
        return tuple(
            sum((d_BB[j, i] if transpose else d_BB[i, j]) * q[j] for j in range(width))
            for i in range(width)
        )

    if kind == "p_ortho":
        rotated = multiply(u)
        return torch.stack(
            multiply(
                tuple(rotated[i] / (1 + alpha * a[i]) for i in range(width)), True
            ),
            -1,
        )
    if kind == "p_direct":
        lower = {}
        for i in range(width):
            for j in range(i + 1):
                value_N = float(i == j) + alpha * sum(
                    d_BB[k, i] * a[k] * d_BB[k, j] for k in range(width)
                )
                for k in range(j):
                    value_N = value_N - lower[i, k] * lower[j, k]
                lower[i, j] = value_N.sqrt() if i == j else value_N / lower[j, j]
        intermediate = []
        for i in range(width):
            value_N = u[i]
            for j in range(i):
                value_N = value_N - lower[i, j] * intermediate[j]
            intermediate.append(value_N / lower[i, i])
        solution = {}
        for i in reversed(range(width)):
            value_N = intermediate[i]
            for j in range(i + 1, width):
                value_N = value_N - lower[j, i] * solution[j]
            solution[i] = value_N / lower[i, i]
        return torch.stack(tuple(solution[i] for i in range(width)), -1)
    if kind not in ("p_k1", "p_k2", "p_k4", "p_k8"):
        raise ValueError("Unknown local operator")
    q = u
    for _ in range(int(kind[3:])):
        dq = multiply(q)
        correction = multiply(tuple(a[i] * dq[i] for i in range(width)), True)
        q = tuple(
            q[i] + 0.5 * (u[i] - q[i] - alpha * correction[i]) for i in range(width)
        )
    return torch.stack(q, -1)


_compiled_apply = None
_compiled_apply_dynamic = None


def apply_compiled(*args, dynamic=False):
    global _compiled_apply, _compiled_apply_dynamic
    if dynamic:
        if _compiled_apply_dynamic is None:
            _compiled_apply_dynamic = torch.compile(
                apply_elementwise, fullgraph=True, dynamic=True
            )
        return _compiled_apply_dynamic(*args)
    if _compiled_apply is None:
        _compiled_apply = torch.compile(
            apply_elementwise, fullgraph=True, dynamic=False
        )
    return _compiled_apply(*args)


def local_forward_operations(kind, width=4):
    """Scoped scalar arithmetic estimate; special functions count as one.

    This describes the coordinate program, not cuSOLVER or GPU instructions.
    Backward is reported as a nominal 2x forward estimate, not an exact VJP count.
    Exact-SVD constraint cost is measured in timings and reported separately.
    """
    if kind == "p_mlp":
        return 4 * width * (width // 2) + 5 * (width // 2) + width
    if kind in ("p_dense", "p_i"):
        return 0
    gate_ops = 5 * width
    if kind == "p_scale":
        return gate_ops + width - 1 + 3 + width
    if kind == "p_diag":
        return gate_ops + 5 * width
    if kind == "p_direct":
        system = width * (width + 1) // 2 * (3 * width + 1)
        cholesky = sum(2 * j + 1 for i in range(width) for j in range(i + 1))
        substitutions = 2 * width * width
        return gate_ops + system + cholesky + substitutions
    if kind.startswith("p_k"):
        return gate_ops + int(kind[3:]) * (4 * width * width + 4 * width)
    raise ValueError("No operation estimate for this control")
