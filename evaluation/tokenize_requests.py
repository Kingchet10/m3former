"""Freeze native SentencePiece likelihood boundaries; no truncation or BOS."""
import gzip, hashlib, json
from pathlib import Path
import sentencepiece as spm

root = Path(__file__).resolve().parent
tok = spm.SentencePieceProcessor(model_file=str(root/'tokenizer/tokenizer.model'))
report = json.loads((root/'active-length-report.json').read_text())
manifest = {}
for key, spec in report['tasks'].items():
    source = root/spec['requests_file']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['requests_sha256']
    dest = root/('tokens-'+source.name.removeprefix('requests-'))
    count = 0
    with gzip.open(source, 'rt') as src, gzip.open(dest, 'wt') as out:
        for line in src:
            row = json.loads(line)
            context, continuation = row['context'], row['continuation']
            # Match harness _encode_pair: transfer trailing context whitespace
            # to the continuation, tokenize concatenation, split by context len.
            spaces = len(context)-len(context.rstrip())
            if spaces:
                continuation = context[-spaces:]+continuation
                context = context[:-spaces]
            ids = tok.encode(context+continuation, out_type=int)
            nctx = len(tok.encode(context, out_type=int))
            assert 0 < nctx < len(ids) <= 4096, (key, row['doc_id'], nctx, len(ids))
            out.write(json.dumps(dict(doc_id=row['doc_id'],idx=row['idx'],ids=ids,nctx=nctx))+'\n')
            count += 1
    assert count == spec['requests']
    refs = root/('references-'+key.rsplit('/',1)[0]+'.json')
    manifest[key] = dict(file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),requests=count,
        references=refs.name,references_sha256=hashlib.sha256(refs.read_bytes()).hexdigest(),
        request_sha256=spec['requests_sha256'],documents=spec['documents'])
    print(key,count,flush=True)
(root/'tokens-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
