# 数据接口与外部资产

不上传 FineWeb-Edu 语料、tokenizer 二进制、benchmark 题目或 checkpoint。`--data-root` 要求已有的冻结数据准备结果，不能直接指向 Hugging Face 数据集名或任意 `.bin` 文件。

Dense 与 MoE 沿用 `megatron_document_pairs_v1`：读取 int32 Megatron indexed 文档，每个文档产生 `ids[:-1] → ids[1:]`，不预测跨文档边界；attention 使用文档因果隔离。数据顺序以冻结 manifest/index 为准，启动器不会重新采样或 shuffle 语料池。

目录至少包含 `manifest.json`、`DATASET_READY`（manifest 的 SHA256）、manifest 引用的索引/分片及 `tokenizer/tokenizer.model`。准确 schema 见 `dense/torchtitan/experiments/brainfuse_scaling/megatron_stream.py` 和 `moe/data.py`。完整性检查、分片选择、训练/验证划分必须沿用原准备产物；本代码导出没有重新下载或重建数据集。

训练数据来源记录为 FineWeb-Edu sample-350BT 的固定子集；不将此短名视为全部采样/去重 provenance。本次整理没有重新验证上游所有分片或进行 benchmark 去污染审计。

Tokenizer：训练绑定的 TinyLlama SentencePiece，32,000 词；本次直接计算本地历史评测文件的 SHA256 为 `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`。请用自己的文件实测匹配，不仅比较 tokenizer 名字。
