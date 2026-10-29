"""FP32 chunked parallel affine EMA scan, including full-context reverse VJP.

Each program carries state across 128-token parallel scans. Reverse scans carry
adjoints across every chunk. No atomics, no truncation, no detached boundaries.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _compose(a0, b0, a1, b1):
    return a0*a1, b0*a1+b1

@tr.jit
def _forward(X, R, RESET, S, T:tl.constexpr, H:tl.constexpr, K:tl.constexpr, C:tl.constexpr=128, F:tl.constexpr=16):
    batch = tl.program_id(0)
    j = tl.program_id(1)*F + tl.arange(0,F)
    rho = tl.load(R+j, j<H*K, 0)
    prev = tl.full((F,),0,tl.float32)
    for start in range(tr.cdiv(T,C)):
        t = start*C+tl.arange(0,C)
        reset = tl.load(RESET+batch*T+t, t<T, 1)
        h = tl.load(X+(batch*T+t[:,None])*H+j[None,:]//K, (t[:,None]<T)&(j[None,:]<H*K),0)
        a = tl.where((t[:,None]<T)&(~reset[:,None]),rho[None,:],0.)
        b = (1.-rho[None,:])*h
        aa,bb = tl.associative_scan((a,b),0,_compose)
        state = prev[None,:]*aa+bb
        tl.store(S+(batch*T+t[:,None])*H*K+j[None,:], state,(t[:,None]<T)&(j[None,:]<H*K))
        prev = tl.sum(tl.where(tl.arange(0,C)[:,None] == min(C,T-start*C)-1,state,0.),0)

@tr.jit
def _backward(X,R,RESET,S,DS,DX4,DR4,T:tl.constexpr,H:tl.constexpr,K:tl.constexpr,C:tl.constexpr=128,F:tl.constexpr=16):
    batch=tl.program_id(0)
    j=tl.program_id(1)*F+tl.arange(0,F)
    rho=tl.load(R+j,j<H*K,0)
    carry=tl.full((F,),0,tl.float32)
    for start in range(tr.cdiv(T,C)):
        t=T-1-(start*C+tl.arange(0,C))
        reset_next=tl.load(RESET+batch*T+t+1,(t>=0)&(t+1<T),1)
        a=tl.where((t[:,None]>=0)&(~reset_next[:,None]),rho[None,:],0.)
        d=tl.load(DS+(batch*T+t[:,None])*H*K+j[None,:],(t[:,None]>=0)&(j[None,:]<H*K),0)
        aa,bb=tl.associative_scan((a,d),0,_compose)
        adj=carry[None,:]*aa+bb
        reset=tl.load(RESET+batch*T+t,t>=0,1)
        prev=tl.load(S+(batch*T+t[:,None]-1)*H*K+j[None,:],(t[:,None]>0)&(~reset[:,None])&(j[None,:]<H*K),0)
        h=tl.load(X+(batch*T+t[:,None])*H+j[None,:]//K,(t[:,None]>=0)&(j[None,:]<H*K),0)
        tl.store(DX4+(batch*T+t[:,None])*H*K+j[None,:],adj*(1.-rho[None,:]),(t[:,None]>=0)&(j[None,:]<H*K))
        tl.store(DR4+(batch*T+t[:,None])*H*K+j[None,:],adj*(prev-h),(t[:,None]>=0)&(j[None,:]<H*K))
        carry=tl.sum(tl.where(tl.arange(0,C)[:,None]==min(C,T-start*C)-1,adj,0.),0)

class _EMA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,rho,reset):
        h,rho,reset=h.contiguous(),rho.contiguous(),reset.contiguous()
        b,t,n=h.shape
        k=rho.shape[-1]
        s=torch.empty((b,t,n,k),device=h.device,dtype=torch.float32)
        _forward[(b,tr.cdiv(n*k,16))](h,rho,reset,s,t,n,k,enable_fp_fusion=False)
        ctx.save_for_backward(h,rho,reset,s)
        return s

    @staticmethod
    def backward(ctx,ds):
        h,rho,reset,s=ctx.saved_tensors
        b,t,n=h.shape
        k=rho.shape[-1]
        dx4=torch.empty_like(s);dr4=torch.empty_like(s)
        _backward[(b,tr.cdiv(n*k,16))](h,rho,reset,s,ds.contiguous(),dx4,dr4,t,n,k,enable_fp_fusion=False)
        return dx4.sum(-1),dr4.sum((0,1)),None

ema=_EMA.apply
