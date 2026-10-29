"""Native-model likelihood evaluation of frozen requests on one GPU."""
import argparse, contextlib, datetime, gzip, hashlib, json, math, os, time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024**2),b''): h.update(chunk)
    return h.hexdigest()

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--check-only',action='store_true');args=p.parse_args()
    root=Path(args.root); spec=json.loads((root/'execution-models.json').read_text())[args.model]
    out=root/'results'/args.model;out.mkdir(parents=True,exist_ok=True)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ['TOKENIZERS_PARALLELISM']='false'
    checkpoint=Path(spec['checkpoint']);weights={}
    audit={'at':stamp(),'checkpoint':str(checkpoint),'source':spec['source'],'family':spec['family'],
           'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
           'dtype':'training-matched parameter precision; BF16 compute; FP32 log-softmax',
           'max_length':4096,'add_bos':False,'rope_change':False,'shard':args.shard,'shards':args.shards}
    original_inner=[]
    if spec['family']=='dense':
        cfg=json.loads((checkpoint.parent.parent/'config.json').read_text())
        from torchtitan.experiments.brainfuse_scaling.models import Architecture,model_config
        a=Architecture(**cfg['architecture']);mc=model_config(a)
        # Build exactly the trained residual model, then switch only the causal
        # attention backend for independent un-packed likelihood sequences.
        with torch.device('cuda'): model=mc.build()
        model.init_weights()
        index=json.loads((checkpoint/'pytorch_model.bin.index.json').read_text())
        proof=json.loads((checkpoint/'model-only.json').read_text())
        for file in sorted(set(index['weight_map'].values())):
            assert sha(checkpoint/file)==proof['files'][file]['sha256'],file
            state=torch.load(checkpoint/file,map_location='cpu',weights_only=True)
            assert not weights.keys() & state.keys()
            weights.update(state)
        assert set(weights)==set(index['weight_map'])
        model.load_state_dict(weights,strict=True);del weights
        assert cfg['training']['mixed_precision_param']=='bfloat16'
        # Match original TorchTitan FSDP forward parameter materialization.
        # Cast parameters only, preserving FP32/complex rotary buffers exactly.
        with torch.no_grad():
            for parameter in model.parameters():parameter.data=parameter.data.to(torch.bfloat16)
        audit['parameter_dtype']='bfloat16, matching saved mixed_precision_param'
        audit['checkpoint_index_sha256']=sha(checkpoint/'pytorch_model.bin.index.json')
        audit['training_architecture']=cfg['architecture']
        model.eval()
        def forward(x):
            pos=torch.arange(x.shape[1],device=x.device)[None,:].expand_as(x)
            return model(x,positions=pos,attention_masks=None)
        # Save original modules for the causal-backend parity check below.
        for layer in model.layers.values(): original_inner.append((layer.attention,layer.attention.inner_attention))
    else:
        os.environ.update(RR_PACKED='1',RR_FUSION='1',RR_COMPILE_NORM='1')
        import torch.distributed as dist
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        rendezvous=out/f'dist-{args.shard}-{os.getpid()}'
        dist.init_process_group('nccl',init_method='file://'+str(rendezvous),rank=0,world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,expert_model_parallel_size=1)
        model_parallel_cuda_manual_seed(47)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=payload['config']; saved=json.loads((checkpoint.parent/'research-summary.json').read_text())
        assert sha(checkpoint)==saved['checkpoint_sha256']
        from architecture import Architecture,build_model
        architecture=dict(cfg['architecture']);architecture['sequence_length']=4096
        router=cfg['router']
        model=build_model(variant=cfg['variant'],architecture=Architecture(**architecture),
            load_balancing_type=router['load_balancing_type'],aux_loss_coeff=router['aux_loss_coeff'],z_loss_coeff=router['z_loss_coeff']).cuda()
        model.load_state_dict(payload['model'],strict=True);del payload
        # Packed training bypasses this ordinary-sequence metadata lookup.
        # No pipeline input is supplied: PP=1 and pre_process=True.
        model.decoder.input_tensor=None
        model.eval()
        audit['parameter_dtype']='float32, matching original explicit DDP recipe'
        audit['checkpoint_sha256']=saved['checkpoint_sha256'];audit['training_architecture']=cfg['architecture']
        audit['eval_architecture_diff']={'sequence_length':{'training':2048,'evaluation':4096}}
        def forward(x):
            pos=torch.arange(x.shape[1],device=x.device)[None,:].expand_as(x)
            return model(x,pos,None)
    expected_dtype=torch.bfloat16 if spec['family']=='dense' else torch.float32
    assert all(p.dtype==expected_dtype for p in model.parameters())
    audit['worker_sha256']=sha(__file__)
    audit['tokens_manifest_sha256']=sha(root/'tokens-manifest.json')
    audit['parameter_count']=sum(p.numel() for p in model.parameters())
    write(out/f'loaded-{args.shard}.json',audit)
    manifest=json.loads((root/'tokens-manifest.json').read_text())
    # Deterministic real prompt for backend, batching, and long-context checks.
    with gzip.open(root/manifest['PIQA/0']['file'],'rt') as f: example=json.loads(next(f))['ids']
    sample=(example*math.ceil(128/len(example)))[:128]
    x=torch.tensor([sample],device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        if original_inner:
            pos=torch.arange(128,device='cuda')[None,:]
            masks=model.get_attention_masks(pos)
            reference=model(x,positions=pos,attention_masks=masks).float()
            from torchtitan.models.common.attention import ScaledDotProductAttention
            for attn,_ in original_inner:attn.inner_attention=ScaledDotProductAttention.Config().build()
            actual=forward(x).float()
            delta=(actual-reference).abs()
            audit['causal_backend_parity']={'max_abs':delta.max().item(),'mean_abs':delta.mean().item(),
                'relative_l2':((actual-reference).norm()/reference.norm()).item(),
                'original':'native FlexAttention','evaluation':'native causal SDPA'}
            assert audit['causal_backend_parity']['relative_l2']<0.01,audit['causal_backend_parity']
            assert audit['causal_backend_parity']['mean_abs']<0.02,audit['causal_backend_parity']
            del reference,actual,delta
        one=forward(x).float()
        padded=torch.full((2,256),2,device='cuda',dtype=torch.long);padded[:,:128]=x
        two=forward(padded)[0,:128].float();delta=(two-one[0]).abs()
        audit['right_padding_parity']={'mean_abs':delta.mean().item(),'max_abs':delta.max().item(),
            'relative_l2':((two-one[0]).norm()/one[0].norm()).item(),
            'argmax_agreement':(two.argmax(-1)==one[0].argmax(-1)).float().mean().item()}
        if audit['right_padding_parity']['relative_l2']>=0.01:
            # Different GEMM shapes need not be bitwise equivalent in BF16,
            # especially through discrete MoE top-k routing. Test causality
            # without changing the numerical kernel shape; retain the error.
            padded[:,128:]=torch.arange(128,device='cuda')%30000+10
            changed=forward(padded)[0,:128].float()
            same=torch.equal(changed,two)
            audit['fixed_shape_causality']={'future_tokens_changed':128,'prefix_bitwise_equal':same,
                'cross_shape_error_retained':True,'interpretation':'shape-dependent numerical difference, not bitwise padding equivalence'}
            write(out/f'padding-diagnostic-{args.shard}.json',audit)
            assert same,'Changing future tokens affected the causal prefix at fixed shape'
            assert audit['right_padding_parity']['argmax_agreement']>=0.99,audit['right_padding_parity']
            del changed
        del one,two,delta,padded
        long_ids=None
        with gzip.open(root/manifest['RACE-M/5']['file'],'rt') as f:
            for line in f:
                row=json.loads(line)
                if long_ids is None or len(row['ids'])>len(long_ids):long_ids=row['ids']
        y=forward(torch.tensor([long_ids[:-1]],device='cuda'))
        assert y.shape==(1,len(long_ids)-1,32000) and torch.isfinite(y).all()
        audit['long_context_check']={'input_tokens':len(long_ids)-1,'finite':True}
        del y
    audit['status']='passed';write(out/f'adapter-check-{args.shard}.json',audit)
    if args.check_only:return
    bucket=lambda n:next(v for v in [128,256,512,1024,2048,4096] if n<=v)
    started=time.time()
    with torch.inference_mode():
        for panel,info in manifest.items():
            task,shot=panel.split('/');dest=out/f'{task}-{shot}shot-shard{args.shard}.jsonl'
            done=dest.with_suffix('.complete.json')
            if done.exists():continue
            assert sha(root/info['file'])==info['sha256']
            seen=set()
            if dest.exists():
                with dest.open() as f:
                    for line in f:
                        row=json.loads(line);seen.add((row['doc_id'],row['idx']))
            requests=[];expected=0
            with gzip.open(root/info['file'],'rt') as f:
                for line in f:
                    row=json.loads(line)
                    if row['doc_id']%args.shards!=args.shard:continue
                    expected+=1
                    if (row['doc_id'],row['idx']) not in seen: requests.append(row)
            requests.sort(key=lambda r:(bucket(len(r['ids'])-1),len(r['ids']),r['doc_id'],r['idx']))
            i=0;count=len(seen);t0=time.time()
            with dest.open('a',buffering=1) as f:
                while i<len(requests):
                    length=bucket(len(requests[i]['ids'])-1);bs=min(32,max(1,8192//length))
                    end=i
                    while end<min(i+bs,len(requests)) and bucket(len(requests[end]['ids'])-1)==length:end+=1
                    rows=requests[i:end];batch=torch.full((len(rows),length),2,device='cuda',dtype=torch.long)
                    for b,row in enumerate(rows):batch[b,:len(row['ids'])-1]=torch.tensor(row['ids'][:-1],device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=forward(batch)
                    for b,row in enumerate(rows):
                        start=row['nctx']-1;stop=len(row['ids'])-1
                        scores=logits[b,start:stop].float()
                        targets=torch.tensor(row['ids'][row['nctx']:],device='cuda')
                        nll=F.cross_entropy(scores,targets,reduction='none')
                        assert torch.isfinite(nll).all()
                        ll=-nll.double().sum().item();greedy=bool((scores.argmax(-1)==targets).all())
                        f.write(json.dumps(dict(doc_id=row['doc_id'],idx=row['idx'],loglikelihood=ll,is_greedy=greedy,
                            target_tokens=len(targets),input_tokens=stop))+'\n')
                    del logits,batch
                    count+=len(rows);i=end
                    write(out/f'progress-{args.shard}.json',dict(at=stamp(),panel=panel,requests=count,total=expected,
                        elapsed_seconds=time.time()-started,panel_seconds=time.time()-t0))
            assert count==expected
            write(done,dict(status='complete',at=stamp(),requests=count,source_sha256=info['sha256'],scores_sha256=sha(dest)))
    write(out/f'shard-{args.shard}-complete.json',dict(status='complete',at=stamp(),panels=len(manifest),elapsed_seconds=time.time()-started))

if __name__=='__main__': main()
