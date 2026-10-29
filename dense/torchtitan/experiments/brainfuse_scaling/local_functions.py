"""Frozen v4 B2 local functions. Pure tensor reference used by eager and compile.

One parameter vector per layer; no state crosses groups, layers or tokens.
CMAC uses indicator reductions rather than atomic table-gradient scatter.
"""
import functools
import math

import torch
import torch.nn.functional as F

MODELS = ("dense", "narrow", "sum", "normalized_linear", "diagonal_quadratic",
          "full_quadratic", "local_mlp", "pi_sigma2", "pi_sigma4", "chebyshev",
          "safe_rational", "rbf", "cmac", "cmac_mlp")
LF_KINDS = tuple("lf_" + m for m in MODELS[2:])+('lf_safe_rational_4to2',)
COUNTS = dict(sum=0, normalized_linear=2, diagonal_quadratic=2,
              full_quadratic=3, local_mlp=16, pi_sigma2=5, pi_sigma4=9,
              chebyshev=14, safe_rational=18, rbf=16, cmac=64, cmac_mlp=64)
COUNTS['safe_rational_4to2']=44
SIGMA0 = 1.0
# Nominal forward scalar operations/group, special functions count as one.
# Backward is explicitly an estimate, not a derived exact VJP operation count.
FORWARD_OPS = dict(sum=2, normalized_linear=14, diagonal_quadratic=16,
                   full_quadratic=20, local_mlp=63, pi_sigma2=43, pi_sigma4=79,
                   chebyshev=70, safe_rational=54, rbf=75, cmac=170, cmac_mlp=219)
# Same nominal convention as two B2 maps, shared normalization and wider dots.
FORWARD_OPS['safe_rational_4to2']=123


def parameter_count(kind, group_size=2):
    if kind=='safe_rational_4to2':
        if group_size!=4:raise ValueError('Safe-Rational4to2 requires B4')
        return 44
    if group_size == 2:
        return COUNTS[kind]
    if kind != 'safe_rational' or group_size not in (4, 8):
        raise ValueError('Only Safe-rational supports B4/B8')
    return 2 * group_size + 14


def initial_values(kind, group_size=2):
    if kind=='safe_rational_4to2':
        parameter_count(kind,group_size)
        from .safe_rational42 import initial_values as values42
        return values42()
    if group_size != 2:
        parameter_count(kind, group_size)
        scale = math.sqrt(2 / group_size)
        angles = [2 * math.pi * i / group_size for i in range(group_size)]
        rows = [scale * (.6 * math.cos(t) + .2 * math.sin(t)) for t in angles]
        rows += [scale * (-.2 * math.cos(t) + .6 * math.sin(t)) for t in angles]
        return rows + initial_values(kind)[4:]
    n = COUNTS[kind]
    values = [0.08 * math.sin(1.7*(i+1)) for i in range(n)]
    if kind.startswith("pi_sigma"):
        k = int(kind[-1])
        # Nonzero, diverse factors. Each raw row is initially strictly inside ball.
        values = [v for i in range(k) for v in
                  (0.7*math.cos(.4+i*1.1), 0.7*math.sin(.4+i*1.1))] + [0.1]
    if kind in ("chebyshev", "safe_rational"):
        values[:4] = [.6, .2, -.2, .6]
    if kind == "rbf":
        values[:8] = [-.5,-.5, -.5,.5, .5,-.5, .5,.5]
        values[8:12] = [math.log(math.expm1(1.0))]*4
    return values


def normalize(p):
    return p * torch.rsqrt(SIGMA0**2 + p.square().sum(-1, keepdim=True))


def bounded_rows(u):
    # clamp squared norm BEFORE sqrt keeps derivative finite even at u=0.
    return u / u.square().sum(-1, keepdim=True).clamp_min(1.0).sqrt()


def cmac_addresses(z):
    # Four fixed shifted 4x4 grids on [-1,1]^2, clipped at outer cells.
    # Exact internal boundaries belong to the upper bin; no straight-through grad.
    return tuple((torch.floor((z[..., 0]+1)*2 + sx).long().clamp(0,3)*4
                  + torch.floor((z[..., 1]+1)*2 + sy).long().clamp(0,3))
                 for sx,sy in ((0.,0.),(.25,.5),(.5,.25),(.75,.75)))


def correction(z, theta, kind):
    if kind=='safe_rational_4to2':
        from .safe_rational42 import correction as correction42
        return correction42(z,theta)
    if kind == 'safe_rational' and z.shape[-1] != 2:
        bsz = z.shape[-1]
        rows = bounded_rows(theta[:2*bsz].reshape(2, bsz))
        t = F.linear(z, rows)
        a = theta[2*bsz:2*bsz+8].reshape(2, 4)
        b = theta[2*bsz+8:].reshape(2, 3)
        numerator = ((a[:,3]*t+a[:,2])*t+a[:,1])*t+a[:,0]
        denominator = 1+((b[:,2]*t+b[:,1])*t+b[:,0]).square()
        return (numerator/denominator).sum(-1)
    x,y = z.unbind(-1)
    if kind == "sum":
        return torch.zeros_like(x)
    if kind == "normalized_linear":
        return theta[0]*x + theta[1]*y
    if kind == "diagonal_quadratic":
        return theta[0]*x.square() + theta[1]*y.square()
    if kind == "full_quadratic":
        return theta[0]*x.square() + 2*theta[1]*x*y + theta[2]*y.square()
    if kind in ("local_mlp", "cmac_mlp"):
        h = 4 if kind == "local_mlp" else 16
        w = theta[:2*h].reshape(h,2)
        hidden = F.silu(x[...,None]*w[:,0] + y[...,None]*w[:,1] + theta[2*h:3*h])
        return (hidden*theta[3*h:]).sum(-1)
    if kind.startswith("pi_sigma"):
        k = int(kind[-1])
        u = bounded_rows(theta[:2*k].reshape(k,2))
        # Keep >=2-degree terms directly: no subtraction of near-equal constants.
        linear = torch.zeros_like(x)
        higher = torch.zeros_like(x)
        for i in range(k):
            ell = (x*u[i,0] + y*u[i,1])/k
            higher = higher*(1+ell) + linear*ell
            linear = linear+ell
        return theta[-1]*higher
    if kind in ("chebyshev", "safe_rational"):
        u = bounded_rows(theta[:4].reshape(2,2))
        t = x[...,None]*u[:,0] + y[...,None]*u[:,1]
        if kind == "chebyshev":
            a = theta[4:].reshape(2,5)
            b1,b2 = torch.zeros_like(t),torch.zeros_like(t)
            for i in (4,3,2,1):
                b0 = 2*t*b1-b2+a[:,i]
                b2,b1 = b1,b0
            return (a[:,0]+t*b1-b2).sum(-1)
        a = theta[4:12].reshape(2,4)
        b = theta[12:].reshape(2,3)
        numerator = ((a[:,3]*t+a[:,2])*t+a[:,1])*t+a[:,0]
        denominator = 1+((b[:,2]*t+b[:,1])*t+b[:,0]).square()
        return (numerator/denominator).sum(-1)
    if kind == "rbf":
        centers = theta[:8].reshape(4,2)
        scale = F.softplus(theta[8:12])+1e-4
        distance = (x[...,None]-centers[:,0]).square()+(y[...,None]-centers[:,1]).square()
        return (theta[12:]*torch.exp(-scale*distance)).sum(-1)
    if kind == "cmac":
        ids = cmac_addresses(z)
        bins = torch.arange(16,device=z.device)
        return sum(((idx[...,None] == bins).to(theta.dtype)*theta[16*i:16*(i+1)]).sum(-1)
                   for i,idx in enumerate(ids))/4
    raise ValueError(kind)


def apply(p, theta, kind):
    parameter_count(kind, p.shape[-1])
    if kind=='safe_rational_4to2':
        from .safe_rational42 import apply as apply42
        return apply42(p,theta)
    z = normalize(p)
    return (p.sum(-1)+correction(z,theta,kind))/math.sqrt(p.shape[-1])


@functools.lru_cache(None)
def compiler(dynamic=False):
    return torch.compile(apply,fullgraph=True,dynamic=dynamic)
