"""Candidate packed expert GEMMs; FP32 master weights and FP32 weight gradients.

No routing, activation, probability weighting or token capacity changes.
"""
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _weight_grad(X,DY,OFF,DW,I:tl.constexpr,O:tl.constexpr,
                 BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
    tile=tl.program_id(0);expert=tl.program_id(1)
    om=(tile//tl.cdiv(I,BN))*BM+tl.arange(0,BM)
    ik=(tile%tl.cdiv(I,BN))*BN+tl.arange(0,BN)
    start=tl.load(OFF+expert-1,mask=expert>0,other=0)
    end=tl.load(OFF+expert)
    acc=tl.zeros((BM,BN),tl.float32)
    for off in range(start,end,BK):
        rows=off+tl.arange(0,BK)
        a=tl.load(DY+rows[None,:]*O+om[:,None],mask=(rows[None,:]<end)&(om[:,None]<O),other=0)
        b=tl.load(X+rows[:,None]*I+ik[None,:],mask=(rows[:,None]<end)&(ik[None,:]<I),other=0)
        acc=tl.dot(a,b,acc)
    tl.store(DW+expert*O*I+om[:,None]*I+ik[None,:],acc,mask=(om[:,None]<O)&(ik[None,:]<I))


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, splits):
        offsets=torch.tensor(splits,device=x.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
        w=weight.to(x.dtype)
        ctx.save_for_backward(x,w,offsets)
        return torch._grouped_mm(x,w.transpose(-1,-2),offs=offsets)

    @staticmethod
    def backward(ctx, dy):
        x,w,offsets=ctx.saved_tensors
        dy=dy.contiguous()
        dx=torch._grouped_mm(dy,w,offs=offsets)
        dw=torch.empty(w.shape,device=w.device,dtype=torch.float32)
        _weight_grad[(triton.cdiv(w.shape[1],32)*triton.cdiv(w.shape[2],64),w.shape[0])](
            x,dy,offsets,dw,w.shape[2],w.shape[1])
        return dx,dw,None


class PackedLinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight=nn.Parameter(torch.stack([getattr(original,f'weight{i}').detach()
                                            for i in range(original.num_gemms)]))

    def forward(self, x, m_splits, **kwargs):
        return GroupedMatmul.apply(x,self.weight,m_splits),None


def replace_experts(model):
    for layer in model.decoder.layers:
        experts=layer.mlp.experts
        experts.linear_fc1=PackedLinear(experts.linear_fc1)
        experts.linear_fc2=PackedLinear(experts.linear_fc2)


if __name__=='__main__':
    torch.manual_seed(47)
    results=[]
    for splits in ([13,27,19],[16,32,16],[0,32,0,16]):
        x=torch.randn(sum(splits),64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(len(splits),96,64,device='cuda',dtype=torch.float32,requires_grad=True)
        y=GroupedMatmul.apply(x,w,splits)
        ref=torch.cat([torch.nn.functional.linear(a,b.to(x.dtype)) for a,b in zip(x.split(splits),w)])
        go=torch.randn_like(y)
        dx,dw=torch.autograd.grad(y,(x,w),go)
        dxr,dwr=torch.autograd.grad(ref,(x,w),go)
        # Reference weight gradients above are BF16-rounded; compare FP32 GEMM oracle too.
        dwf=torch.stack([g.float().T@a.float() for a,g in zip(x.split(splits),go.split(splits))])
        torch.testing.assert_close(y,ref,atol=.125,rtol=.02)
        torch.testing.assert_close(dx,dxr,atol=.125,rtol=.02)
        torch.testing.assert_close(dw,dwf,atol=.003,rtol=.002)
        results.append(dict(splits=splits,output_max=(y-ref).abs().max().item(),
            dx_max=(dx-dxr).abs().max().item(),dw_fp32_max=(dw-dwf).abs().max().item()))
    print(results)
