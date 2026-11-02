# Data

The trainers read a corpus directory in the `megatron_document_pairs_v1` format:

```text
corpus/
  manifest.json          splits, file sizes and SHA-256 hashes, tokenizer, artifact_id
  DATASET_READY          SHA-256 of manifest.json
  tokenizer/tokenizer.model
  train.bin, train.idx   int32 Megatron indexed dataset, one sequence per document
  train.pairs.i64        cumulative number of next-token pairs per document
  validation.bin, validation.idx, validation.pairs.i64
```

Each document of length n contributes the n - 1 pairs (ids[:-1], ids[1:]); a model never predicts across a document
boundary, and positions restart at every document. Training cuts the pair stream into sequences of 2,048 tokens in
order, so the data order is fixed by the corpus. `scripts/prepare_data.py` builds such a directory from a Hugging Face
dataset or a JSON-lines file; the readers are `dense/m3former/megatron_stream.py` and `moe/data.py`.

The corpus of the paper is a fixed sample of FineWeb-Edu (sample-350BT) tokenized with the TinyLlama SentencePiece
tokenizer (32,000 tokens; SHA-256 of `tokenizer.model`:
`9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`), with 99,876,864 validation label tokens.
`prepare_data.py` reproduces its format and split sizes, not its exact documents.

Token budgets: a model with N non-embedding parameters trains on 20N tokens (5N and 10N for the branches); the
training split must hold at least 20N label tokens of the largest dense model you train (27.3B for 1.4B). The MoE
recipes train on 20 tokens per active non-embedding parameter (6.2B and 15.9B tokens).
