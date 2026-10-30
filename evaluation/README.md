# Downstream evaluation

Zero- and five-shot accuracy on LAMBADA, PIQA, WinoGrande, ARC-Easy, ARC-Challenge, SciQ, LogiQA, BoolQ, HellaSwag
and RACE (middle school), scored by log-likelihood with the native models of this repository.

The requests (contexts and candidate continuations) were generated with the LM Evaluation Harness at commit
`d6de81643928d653435c431bae19945d41d32520`, with the dataset revisions in `protocol/dataset-revisions.json` and fixed
few-shot examples (seed 47). `tokenize_requests.py` tokenizes them with the TinyLlama tokenizer exactly as the
harness splits context and continuation, without BOS and without truncation (the longest prompt plus answer has
2,557 tokens; the models run at a context of 4,096 without changing RoPE). The benchmark texts themselves are not
redistributed here.

Scoring:

1. Export a dense checkpoint with `scripts/export_dense_checkpoint.py` (MoE runs save `final.pt` directly).
2. Prepare an evaluation directory with `execution-models.json` (for each model: `family` dense or moe, `source`,
   `checkpoint`), `tokens-manifest.json`, the tokenized requests and the references of every task.
3. Run `worker.py --root EVAL_ROOT --model NAME --shard i --shards 4` for the four shards, with `PYTHONPATH` set to
   `dense/` (dense) or `moe/` and `third_party/Megatron-LM` (MoE).
4. Run `aggregate.py --root EVAL_ROOT --model NAME` after all four shards are complete.

The worker checks the weight hashes, loads the state dict strictly, checks causality and finite outputs, keeps the
raw candidate scores and resumes completed requests. PIQA, ARC-E, ARC-C and HellaSwag use length-normalized accuracy,
the other tasks accuracy; the average is the unweighted mean of the ten tasks.
