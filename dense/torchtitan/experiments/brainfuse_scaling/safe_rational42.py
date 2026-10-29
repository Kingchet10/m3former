"""Four-input/two-output Safe-Rational; fixed pairs only in the linear skip."""
import math
import torch
import torch.nn.functional as F


def initial_values():
    from .local_functions import initial_values as pair_values
    tail=pair_values('safe_rational')[4:]
    return ([.6,.2,0.,0., -.2,.6,0.,0.]+tail+
            [0.,0.,.6,.2, 0.,0.,-.2,.6]+tail)


def correction(z,theta):
    from .local_functions import bounded_rows
    a=theta.reshape(2,22)
    rows=bounded_rows(a[:,:8].reshape(2,2,4))
    t=F.linear(z,rows.flatten(0,1)).unflatten(-1,(2,2))
    num=a[:,8:16].reshape(2,2,4)
    den=a[:,16:22].reshape(2,2,3)
    numerator=((num[...,3]*t+num[...,2])*t+num[...,1])*t+num[...,0]
    denominator=1+((den[...,2]*t+den[...,1])*t+den[...,0]).square()
    return (numerator/denominator).sum(-1)


def apply(p,theta):
    if p.shape[-1]!=4 or theta.numel()!=44:
        raise ValueError('Safe-Rational4to2 requires four inputs and 44 layer-shared parameters')
    z=p*torch.rsqrt(1+p.square().sum(-1,keepdim=True))
    skip=p.unflatten(-1,(2,2)).sum(-1)
    return (skip+correction(z,theta))/math.sqrt(2)
