"""Token-local ridge branch. H=input, R=ridge, N=compressed channels."""
import functools
import torch
import torch.nn.functional as F
from .local_functions import SIGMA0, initial_values


def coefficients():
    old = initial_values('safe_rational')
    return old[4:8] + old[12:15]


def rational(t_R, theta):
    a, b = theta[:4], theta[4:]
    return ((a[3]*t_R+a[2])*t_R+a[1])*t_R+a[0], 1+((b[2]*t_R+b[1])*t_R+b[0]).square()


def code(q_R, theta, kind):
    if kind == 'linear':
        return q_R
    if kind == 'silu':
        return F.silu(q_R)
    if kind != 'rational':
        raise ValueError(kind)
    t_R = q_R * torch.rsqrt(SIGMA0**2+q_R.square())
    numerator_R, denominator_R = rational(t_R, theta)
    return numerator_R / denominator_R


def branch(p_H, v_RH, u_NR, theta, gamma, kind):
    q_R = F.linear(p_H, v_RH)
    g_R = code(q_R, theta, kind)
    return gamma * F.linear(g_R, u_NR)


@functools.lru_cache(None)
def compiler(dynamic=False):
    return torch.compile(branch, fullgraph=True, dynamic=dynamic)


def derivative(t_R, theta):
    a, b = theta[:4], theta[4:]
    numerator_R, denominator_R = rational(t_R, theta)
    q_R = (b[2]*t_R+b[1])*t_R+b[0]
    dp_R = (3*a[3]*t_R+2*a[2])*t_R+a[1]
    dq_R = 2*b[2]*t_R+b[1]
    return (dp_R*denominator_R-2*numerator_R*q_R*dq_R)/denominator_R.square()
