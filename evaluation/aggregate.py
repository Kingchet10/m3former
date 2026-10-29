import argparse,csv,fcntl,json,math
from pathlib import Path
from worker import write,sha,stamp
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True);a=p.parse_args()
root=Path(a.root);out=root/'results'/a.model;manifest=json.loads((root/'tokens-manifest.json').read_text());result={}
for panel,info in manifest.items():
    task,shot=panel.split('/');docs={};total=0
    refs=json.loads((root/info['references']).read_text());assert sha(root/info['references'])==info['references_sha256']
    for shard in range(4):
        path=out/f'{task}-{shot}shot-shard{shard}.jsonl';receipt=json.loads(path.with_suffix('.complete.json').read_text())
        assert receipt['status']=='complete' and receipt['scores_sha256']==sha(path) and receipt['source_sha256']==info['sha256']
        for line in path.read_text().splitlines():
            row=json.loads(line);d=docs.setdefault(row['doc_id'],{});assert row['idx'] not in d;d[row['idx']]=row;total+=1
    assert len(docs)==info['documents']==len(refs) and total==info['requests']
    acc=[];norm=[]
    perdoc=[]
    for ref in refs:
        values=docs[ref['doc_id']];assert sorted(values)==list(range(len(values)))
        if ref['kind']=='loglikelihood':correct=int(values[0]['is_greedy']);normalized=None
        else:
            scores=[values[i]['loglikelihood'] for i in range(len(values))]
            assert len(scores)==len(ref['choice_lengths'])
            pred=max(range(len(scores)),key=lambda i:scores[i]);pn=max(range(len(scores)),key=lambda i:scores[i]/ref['choice_lengths'][i])
            correct=int(pred==ref['gold']);normalized=int(pn==ref['gold']);norm.append(normalized)
        acc.append(correct);perdoc.append(dict(doc_id=ref['doc_id'],acc=correct,acc_norm=normalized))
    use_norm=task in ('PIQA','ARC-E','ARC-C','HellaSwag')
    chosen=norm if use_norm else acc;mean=sum(chosen)/len(chosen)
    result[panel]=dict(documents=len(acc),acc=sum(acc)/len(acc),acc_norm=sum(norm)/len(norm) if norm else None,
        table_metric='acc_norm' if use_norm else 'acc',table_score=100*mean,
        standard_error=100*math.sqrt(mean*(1-mean)/(len(chosen)-1)))
    write(out/f'{task}-{shot}shot-doc-results.json',perdoc)
averages={str(shot):sum(v['table_score'] for k,v in result.items() if k.endswith('/'+str(shot)))/10 for shot in (0,5)}
write(out/'results.json',dict(status='complete',at=stamp(),model=a.model,panels=result,average=averages,
    model_binding=json.loads((root/'execution-models.json').read_text())[a.model],tokens_manifest_sha256=sha(root/'tokens-manifest.json'),
    max_length=4096,shots=[0,5],excluded_tasks=['RACE-H'],fewshot_seed=47))
print(json.dumps(averages),flush=True)
with (root/'summary.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    all_models=json.loads((root/'models.json').read_text())['models']
    completed={}
    for item in all_models:
        p=root/'results'/item['run']/'results.json'
        if p.exists():completed[item['run']]=json.loads(p.read_text())
    tasks=list(dict.fromkeys(k.split('/')[0] for k in manifest))
    for shot in (0,5):
        dest=root/f'results-{shot}shot.csv';temp=dest.with_suffix('.csv.tmp')
        with temp.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['run','status','PPL',*tasks,'Avg'])
            for item in all_models:
                name=item['run'];r=completed.get(name)
                writer.writerow([name,'complete' if r else 'pending',item.get('ppl',math.exp(item['loss'])),
                    *[r['panels'][t+'/'+str(shot)]['table_score'] if r else '' for t in tasks],
                    r['average'][str(shot)] if r else ''])
        temp.replace(dest)
    write(root/'summary.json',dict(status='complete' if len(completed)==len(all_models) else 'partial',
        at=stamp(),models_complete=len(completed),models_total=len(all_models),
        panels_complete=sum(len(v['panels']) for v in completed.values()),panels_total=280,
        completed=list(completed),pending=[x['run'] for x in all_models if x['run'] not in completed]))
