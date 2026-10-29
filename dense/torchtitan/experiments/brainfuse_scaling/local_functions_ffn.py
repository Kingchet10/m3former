"""Production FeedForward adapter for the v4 local-function screen."""
from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.feed_forward import FeedForward
from .local_functions import COUNTS, apply, compiler, correction, initial_values, normalize, parameter_count


class LocalFunctionFeedForward(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        kind: str = "sum"
        local_compile: bool = True
        group_size: int = 2
        dpr_shifts: int = 0
        ridge_rank: int = 0
        ridge_kind: str = 'rational'
        ridge_seed: int = 0
        ridge_layer: int = 0
        ridge_v_std: float = .02
        ridge_u_std: float = .02

        def build(self, **kwargs):
            if self.sharding_config is not None and COUNTS[self.kind]:
                import spmd_types as spmd
                from torchtitan.models.common.decoder_sharding import dense_param_placement
                self.sharding_config = replace(self.sharding_config, state_shardings={
                    **self.sharding_config.state_shardings,
                    "local_theta":dense_param_placement(tp=spmd.R),
                    **({"shift_weight":dense_param_placement(tp=spmd.R),
                        "shift_bias":dense_param_placement(tp=spmd.R)} if self.dpr_shifts else {}),
                    **({name:dense_param_placement(tp=spmd.R) for name in
                        ('ridge_v','ridge_u','ridge_gamma',*(['ridge_theta'] if self.ridge_kind=='rational' else []))}
                       if self.ridge_rank else {})})
            return FeedForward.Config.build(self, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        self.group_size = config.group_size
        self.outputs_per_group=2 if config.kind=='safe_rational_4to2' else 1
        count = parameter_count(config.kind, self.group_size)
        if config.w1.out_features % self.group_size or config.w2.in_features*self.group_size != config.w1.out_features*self.outputs_per_group:
            raise ValueError('Local function requires H divisible by B and N=H/B')
        self.kind,self.local_compile = config.kind,config.local_compile
        if COUNTS[self.kind]:
            self.local_theta = nn.Parameter(torch.empty(count))
        else:
            self.register_parameter("local_theta",None)
        self.capture_diagnostics,self.diagnostics = False,{}
        self.dpr_shifts = config.dpr_shifts
        if self.dpr_shifts:
            if self.kind != 'safe_rational' or self.dpr_shifts not in (1,4) or self.group_size != 2:
                raise ValueError('DPR requires Safe-rational and 1/4 shifts')
            if config.w1.out_features != 1024 or config.w1.out_features % (2*self.dpr_shifts):
                raise ValueError('Frozen DPR requires H1024 and equal stream blocks')
            self.shift_weight = nn.Parameter(torch.empty(self.dpr_shifts,config.w1.in_features))
            self.shift_bias = nn.Parameter(torch.empty(self.dpr_shifts))
        self.shift_mode,self.shift_mean,self.shift_offset = 'normal',None,1
        self.capture_shifts,self.last_shifts = False,None
        self.ridge_rank,self.ridge_kind = config.ridge_rank,config.ridge_kind
        if self.ridge_rank:
            if self.kind!='safe_rational' or self.dpr_shifts or self.ridge_rank not in (4,8,16):
                raise ValueError('RRM requires Safe-rational and rank4/8/16')
            self.ridge_v=nn.Parameter(torch.empty(self.ridge_rank,config.w1.out_features))
            self.ridge_u=nn.Parameter(torch.empty(config.w2.in_features,self.ridge_rank))
            self.ridge_gamma=nn.Parameter(torch.empty(1))
            self.ridge_theta=nn.Parameter(torch.empty(7)) if self.ridge_kind=='rational' else None
            self.ridge_init=(config.ridge_seed,config.ridge_layer,config.ridge_v_std,config.ridge_u_std)
        self.ridge_mode='normal'
        self.capture_ridge=False
        self.ridge_offset=1
        self.ridge_mean=self.ridge_affine=self.last_ridge=None

    def reset_parameters(self):
        if self.local_theta is not None:
            with torch.no_grad():
                for i,v in enumerate(initial_values(self.kind, self.group_size)):
                    self.local_theta[i].fill_(v)
        if self.dpr_shifts:
            with torch.no_grad():
                self.shift_weight.zero_()
                self.shift_bias.zero_()
        if self.ridge_rank:
            from functools import partial
            from .paired_init import PairedInitializer
            from .rational_ridge import coefficients
            seed,layer,v_std,u_std=self.ridge_init
            for name,std in [('ridge_v',v_std),('ridge_u',u_std)]:
                PairedInitializer(partial(nn.init.trunc_normal_,std=std),seed,
                                  f'layers.{layer}.feed_forward.{name}')(getattr(self,name))
            with torch.no_grad():
                self.ridge_gamma.zero_()
                if self.ridge_theta is not None:
                    for i,value in enumerate(coefficients()):self.ridge_theta[i].fill_(value)

    def forward(self,x):
        p = F.silu(self.w1(x))*self.w3(x)
        with torch.autocast(device_type=p.device.type,enabled=False):
            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            u = p.to(dtype).unflatten(-1,(-1,self.group_size))
            theta = self.local_theta.to(dtype) if self.local_theta is not None else u.new_empty(0)
            fn = compiler(dynamic=not self.training) if self.local_compile else apply
            if self.dpr_shifts:
                from .dynamic_pair import shifts,compress,compiler as dpr_compiler
                delta = shifts(x.to(dtype),self.shift_weight.to(dtype),self.shift_bias.to(dtype),
                               p.shape[-1]//(2*self.dpr_shifts))
                if self.capture_shifts:self.last_shifts = delta.detach().cpu()
                if self.shift_mode != 'normal' and self.training:
                    raise RuntimeError('Shift interventions are evaluation-only')
                if self.shift_mode == 'zero':delta = torch.zeros_like(delta)
                elif self.shift_mode == 'mean':delta = torch.zeros_like(delta)+self.shift_mean.to(delta)
                elif self.shift_mode == 'shuffle':
                    delta = torch.roll(delta.reshape(-1,self.dpr_shifts),self.shift_offset,0).reshape_as(delta)
                elif self.shift_mode not in ('normal','rounded'):raise ValueError(self.shift_mode)
                dpr_fn = dpr_compiler(dynamic=not self.training) if self.local_compile else compress
                y = dpr_fn(p.to(dtype),delta,theta,hard=self.shift_mode=='rounded')
            else:
                y = fn(u,theta,self.kind)
            if self.ridge_rank:
                from .rational_ridge import branch,compiler as ridge_compiler,code
                p_H=p.to(dtype);v_RH=self.ridge_v.to(dtype);u_NR=self.ridge_u.to(dtype)
                theta_R=self.ridge_theta.to(dtype) if self.ridge_theta is not None else p_H.new_empty(0)
                gamma=self.ridge_gamma.to(dtype)
                if self.ridge_mode!='normal' and self.training:
                    raise RuntimeError('Ridge interventions are evaluation-only')
                if self.ridge_mode=='normal':
                    ridge_fn=ridge_compiler(dynamic=not self.training) if self.local_compile else branch
                    c_N=ridge_fn(p_H,v_RH,u_NR,theta_R,gamma,self.ridge_kind)
                else:
                    q_R=F.linear(p_H,v_RH);g_R=code(q_R,theta_R,self.ridge_kind)
                    if self.ridge_mode=='off':g_R=torch.zeros_like(g_R)
                    elif self.ridge_mode=='mean':g_R=torch.zeros_like(g_R)+self.ridge_mean.to(g_R)
                    elif self.ridge_mode=='shuffle':g_R=torch.roll(g_R.reshape(-1,self.ridge_rank),self.ridge_offset,0).reshape_as(g_R)
                    elif self.ridge_mode=='affine':g_R=q_R*self.ridge_affine[0].to(q_R)+self.ridge_affine[1].to(q_R)
                    else:raise ValueError(self.ridge_mode)
                    c_N=gamma*F.linear(g_R,u_NR)
                if self.capture_ridge:
                    with torch.no_grad():
                        q_R=F.linear(p_H,v_RH);g_R=code(q_R,theta_R,self.ridge_kind)
                        self.last_ridge=(q_R.detach().cpu(),g_R.detach().cpu())
                if self.capture_diagnostics:
                    with torch.no_grad():
                        self.ridge_diagnostics=dict(ridge_gamma=gamma.detach(),ridge_output_rms=c_N.square().mean().sqrt(),
                            ridge_to_main_rms=c_N.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))
                y=y+c_N
            if self.capture_diagnostics:
                with torch.no_grad():
                    c = correction(normalize(u),theta,self.kind)
                    self.diagnostics = dict(local_output_rms=y.square().mean().sqrt(),
                        local_output_abs_max=y.abs().max(),correction_rms=c.square().mean().sqrt(),
                        input_rms=u.square().mean().sqrt(),
                        local_parameter_norm=theta.norm())
                    if self.dpr_shifts:
                        self.diagnostics['unshifted_correction_rms'] = self.diagnostics.pop('correction_rms')
                        self.diagnostics.update(shift_abs_mean=delta.abs().mean(),shift_abs_max=delta.abs().max())
                    if self.ridge_rank:self.diagnostics.update(self.ridge_diagnostics)
        if self.outputs_per_group==2:y=y.flatten(-2)
        return self.w2(y.to(p.dtype))
