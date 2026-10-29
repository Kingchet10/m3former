# 整理来源与改动边界

导出日期：2026-10-29。独立目录生成，没有修改或覆盖原始实验目录。

1. Dense 从 `dense-attnres-ladder-20261026/source` 的实际封存工作树收集依赖闭包。该工作树 Git HEAD 为 `87f2ea945cda714e568f381fb771f5e842ae9088`，包含未提交的实验更改；这个 SHA 不能单独标识导出，也不能在公开 origin 找到。实际文件 SHA256 见 `provenance/source-files.json`。
2. WSD 审计修复来自同 campaign 的 `rng_fix/rr_wsd_rng_fix`；model-only 导出来自其 `code/retain.py`。
3. MoE 核心与运行时来自 `large-a09b-20b-20261024`。小/大档正式 JSON 分别来自 `optimized-256k/configs` 和 `large-a09b-20b/configs`。这些文件保留原科学设置，只有外部路径占位化。
4. Dense 的 66 份 recipe 是本地保存的解析结果；`provenance/configs.json` 记录原文件 SHA。它们覆盖正式的五规模三预算四方法以及 700M dd40/dd80；重复的默认对象 repr 是原配置记录的一部分，不代表新运行的内存地址。
5. 新增 `scripts/launch.py` 只展开可配置的数据/运行路径与 Python 模块搜索路径，保持配方中 GPU 数、GBS、LR、预算不变；5N/10N 强制使用已经部署过的 RNG-preserving 分支审计入口。
6. FLA 两个命名空间只删除无关 eager imports，避免复制整个算法库；原/新哈希均保留。训练算子、forward/backward 未编辑。
7. 下游结果来自 2026-10-28 完成的 14 模型、280 面板评测；导出的评分器包含当时的大 MoE 数值检查修复。报告跨 shape BF16 数值差异，不声称 bitwise 一致。
8. 小 MoE 的 `architecture.py` / `residuals.py` 单独保存在 `moe/variants/small/`，按原 source-manifest 核验哈希；它们与大档的组数及支持的 variant 名不同。启动器按 1024/1536 宽度选择对应模块，未改动两个正式计算图。
9. Dense 新快照比早期配方多两个关闭的字段 `hc_streams=0`、`attnres_block_size=0`；配置对照仅允许这两个明确的兼容默认值，不忽略其他架构差异。TensorBoard 大 FLOP 步数钩子保留在 `dense/runtime_hooks/`，只钳制事件索引，精确计数仍在 JSON 日志中。

`provenance/SHA256SUMS.json` 是整个导出包的文件清单，`validation.json` 是本次整理验证结果。新 GPU 训练与跨环境数值等价不在本次已验证范围。
