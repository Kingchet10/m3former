# Frozen-request downstream scorer

Included: original `worker.py`, `aggregate.py`, and `tokenize_requests.py`, plus protocol metadata and completed result CSVs. Excluded: benchmark prompt/answer text, tokenizer binary, model weights, absolute cluster bindings, and scheduler/controller scripts.

`worker.py --root EVAL_ROOT --model RUN --shard 0 --shards 4` expects the original frozen-evaluation directory contract: `execution-models.json`, `models.json`, `tokens-manifest.json`, per-task tokenized request `.jsonl.gz`, per-task references and checkpoint files. `execution-models.json` binds family, source and checkpoint. Set PYTHONPATH to this repository's dense or MoE framework before launching each worker. This export does not automatically generate those external assets.

Use four shards for the unchanged `aggregate.py`; only aggregate after all four completion receipts exist. The scorer checks weight hashes, strict state-dict loading, finite outputs and causality, retains raw candidate scores, and resumes completed request IDs. Evaluation is not training.

Protocol: 10 tasks, 0/5-shot, fixed demonstrations seed 47, no RACE-H. LAMBADA standard; RACE middle per question. PIQA/ARC-E/ARC-C/HellaSwag use acc_norm; others acc; Avg is ten-task equal mean, excluding PPL. Context cap 4096, no prompt truncation and no RoPE changes. Some 5-shot prompts exceed training context 2048; maximum complete prompt/candidate length 2557. No near-duplicate decontamination claim.

Large MoE cross-shape BF16 output differences were retained in diagnostics. Fixed-shape future-token perturbation checks passed; this is not a claim that logits or all benchmark predictions are invariant to batch shape. See the worker's checks and historical documentation for the boundary.
