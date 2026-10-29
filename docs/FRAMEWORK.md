# 框架与计算图导航

## Dense / TorchTitan

`scripts/launch.py` 展开冻结配方 → `torch.distributed.run` → `torchtitan.train` → `ConfigManager` → `config_registry.wsd` / `rr_wsd_rng_fix.config.wsd` → `WSDTrainer` → `ResearchTrainer` → `Architecture` / `model_config`。

模型与训练不混成一个文件：

- `dense/torchtitan/models/common/`：attention、QKV 投影、RMSNorm、SwiGLU、RoPE、通用 decoder。
- `dense/torchtitan/models/llama3/`：普通 Transformer、初始化、分布式包装与 state-dict 接口。
- `dense/torchtitan/components/`：优化器、调度、数据接口、checkpoint 等公共设施。
- `dense/torchtitan/distributed/`：实际训练依赖的 DP/FSDP/SPMD 支持。
- `dense/torchtitan/experiments/brainfuse_scaling/`：本研究的方法装配、数据适配、WSD 和统计。

`depth_dendrites.py` 是 MUDD/dd 的第一阅读入口。分组发生在 hidden/channel 维，每组分别生成跨历史深度的混合权重；Q/K/V/R 是四个读取输出。分组混合不表示整个网络组间隔离，QKV、Attention 和 FFN 中的投影仍可跨通道交互。单路分组 SW 与四路 dd 是不同对照。混合的对象是历史状态，不能称为原始 Attention/FFN 增量。

`depth_mix_kernel.py` 提供融合历史混合；`block_attn_residuals.py` 提供块级 Attention Residuals。Dense 配置 `blockattnres8` 的 8 是子层数，等于 4 个 Transformer 层，不是 8 个 Transformer 层。

WSD 的 warmup 固定为 20N 主线步数的 1%；最后每个预算的 20% 使用 `0.1 + 0.9*(1-sqrt(p))`。5N/10N 从主线相应锚点恢复模型、Adam、loader、scheduler、RNG 后独立退火；不重新 warmup。RNG 修复保留恢复断言，只消除审计本身消耗随机数的问题。

## MoE / Megatron Core

`scripts/launch.py` → `moe/train.py` → Megatron parallel state 初始化（TP/PP/EP=1）→ `architecture.build_model` → `GPTModel` → `ResidualDecoder` → DDP 包装。

- `architecture.py` 明确所有骨干和专家设置，关闭参考运行时的可选 Qwen/Loop 路由扩展。
- `residuals.py` 直接使用 Megatron 原 decoder 的 Attention/MLP 模块，改变的是残差读取和接线；history 只在一次 forward 内存在。
- `packed_experts.py` 把专家矩阵打包并调用 `torch._grouped_mm`；自定义 Triton backward 保留 FP32 权重梯度。
- `depth_mix_kernel.py` / `fla_attnres_op.py` 是计算与自动微分接口，不负责调度器、数据或 checkpoint。
- `data.py` 保留文档内相邻 token 预测对；`train.packed` 以 `PackedSeqParams` 显式保持文档与序列边界。

小 MoE：24×1024，64 experts，Top-8，expert FFN=352，baseline/MUDD/dd64/Block AttnRes，约 10B tokens。
大 MoE：28×1536，64 experts，Top-8，expert FFN=512，baseline/dd96，约 20B tokens。
两档均 GBS 128×2048，峰值 LR 3e-4，1% warmup 后 cosine 到 10%；不要误用 Dense 的 GBS64/WSD 配方。

两个正式版本的 dd 组数原先写在各自 `residuals.py` 中。本导出保留大档原文件和 `moe/variants/small/` 的小档原文件，启动器按冻结结构选择 import 路径，避免用 dd96 的构造器装载 dd64 配方。训练器、数据、打包专家和两类融合核在两档之间逐字节相同；共享 cosine 函数公式也相同。不能手动只修改 JSON 的 `residual_groups` 就认为组数生效。

## 下游评测

`evaluation/worker.py` 加载原生 checkpoint，对预先冻结的候选答案做 log-likelihood 评分；`aggregate.py` 检查分片数量/哈希，按固定指标聚合。它们是评分与汇总核心，不是包含所有 benchmark 文本和权重的一键评测包。详见 `evaluation/README.md`。
