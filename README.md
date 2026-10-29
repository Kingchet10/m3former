# Routed Residuals — core research implementation

这是实验代码的独立整理版：重点是**残差计算图如何接入训练框架**，以及同一配方下的 baseline、MUDD、分组 dd/M3、Block Attention Residuals 对照。该仓库不包含训练数据、模型权重、账号凭证或集群提交代码。

## 从这里阅读

| 部分 | 入口 | 内容 |
|---|---|---|
| Dense 方法 | [depth_dendrites.py](dense/torchtitan/experiments/brainfuse_scaling/depth_dendrites.py) | MUDD、分组多路历史混合、单路分组、可选 PrePost |
| Dense 框架接入 | [models.py](dense/torchtitan/experiments/brainfuse_scaling/models.py) | 用 TorchTitan 构建骨干并替换残差计算图 |
| Dense Block AttnRes | [block_attn_residuals.py](dense/torchtitan/experiments/brainfuse_scaling/block_attn_residuals.py) | 层间块状态、块内累积和注意力读取 |
| Dense 训练 | [trainer.py](dense/torchtitan/experiments/brainfuse_scaling/trainer.py) | 训练状态、验证、token 计数及配对初始化 |
| WSD | [wsd_runtime.py](dense/torchtitan/experiments/brainfuse_scaling/wsd_runtime.py) | 稳定主线、预算末端退火、完整分支状态与恢复检查 |
| WSD 恢复修复 | [config.py](dense/rr_wsd_rng_fix/config.py) | 指纹审计不消耗恢复后的随机数状态 |
| MoE 框架接入 | [architecture.py](moe/architecture.py)、[small/architecture.py](moe/variants/small/architecture.py) | Megatron Core 模型与 MoE 配置构造 |
| MoE 残差方法 | [residuals.py](moe/residuals.py)、[small/residuals.py](moe/variants/small/residuals.py) | 大档 dd96 / 小档 dd64，以及 baseline、MUDD、Block AttnRes 的层连接 |
| MoE 训练 | [train.py](moe/train.py) | PyTorch DDP、AdamW、文档隔离、验证与最终权重保存 |
| 速度核心 | [packed_experts.py](moe/packed_experts.py)、[depth_mix_kernel.py](moe/depth_mix_kernel.py) | 打包专家 grouped GEMM、FP32 权重梯度 Triton 核、历史混合核 |
| 下游评测 | [evaluation/](evaluation/) | 原生 checkpoint 似然评分与完整分片聚合 |

更详细的层次及方法边界见 [FRAMEWORK.md](docs/FRAMEWORK.md)。

## 目录

```text
dense/                 TorchTitan 训练所需源码子集、WSD、66 份历史配方
moe/                   Megatron Core 接入、DDP 训练器、6 份正式配方
third_party/           实际使用的 Megatron Core 快照、精简 FLA AttnRes 依赖
scripts/launch.py      可配置数据/输出目录的单节点入口，不申请或调度 GPU
evaluation/            下游评分与聚合核心、固定评测协议元数据
results/               完成的 0/5-shot CSV 和两页论文表格
provenance/            逐文件源码 SHA256、配置来源和整理差异
licenses/              第三方许可证
```

Dense 为 **TorchTitan**。MoE 为 **Megatron Core 模型组件 + 自定义 PyTorch DDP/AdamW 训练循环**；TP=PP=EP=1，各 rank 复制全部专家。它不是 Megatron 原生完整训练引擎，也不支持在此配方中直接开启 TP/PP/EP。

## 环境与运行

Dense 和 MoE 使用不同的训练环境，请分开管理。Dense 历史环境使用 `torch==2.12.0.dev20260408+cu128`，依赖见 [requirements-edu500m.txt](dense/requirements-edu500m.txt)；MoE 历史环境使用 PyTorch 2.9.0、CUDA 12.8、Transformer Engine 2.11.0、FlashAttention 2.8.3、Triton 3.5.0、NumPy 1.26.4，版本直接来自完成训练的 runtime.json，见 [moe/requirements.txt](moe/requirements.txt)。仓库不复制 Python/CUDA 环境，不保证历史 nightly wheel 仍可下载。不要将任意新版本依赖当成已验证等价。

需提供已有的冻结 document-pair 数据目录，详见 [DATA.md](docs/DATA.md)。示例中的路径由使用者填写；运行命令即启动训练，`--dry-run` 仅展示命令。

```bash
# 在 Dense 环境内，100M baseline 的 20N 主线（原配方 4 GPU）
python scripts/launch.py dense \
  --config dense/configs/rr100-base-wsd-20N-1017-edu350-128k-4gpu.json \
  --data-root /path/to/frozen-data --runs-root /path/to/new-runs

# 在 MoE 环境内，小 MoE dd64（原配方 8 GPU）
python scripts/launch.py moe --config moe/configs/small/dd64.json \
  --data-root /path/to/frozen-data --runs-root /path/to/new-runs
```

Dense 5N/10N 配方要求对应 20N 主线保存的完整锚点；传同一个 `--runs-root`。分支共享主线前缀，不从最终权重反推优化器状态。此次整理**没有加入自动队列、等待全部模型完成条件或自动删除 checkpoint**；分支执行和锚点释放由操作者决定。`dense/export_checkpoint.py` 保留历史验证后导出功能，使用前阅读其命令行和输入清单格式。

## 科研与复现边界

- 核心计算源码取自已使用的封存快照，不以重写的教学实现替换正式算子。Dense 以 2026-10-26 AttnRes ladder 的兼容源码为整理基底；不能把这一份快照哈希冒充全部历史运行各自的执行哈希。
- 保留 Dense baseline/MUDD/dd/AttnRes 的五档三预算配置，以及额外 700M 组数配置。1.4B 原 MUDD 曾失稳；配方存在不代表该终点是可用成功结果。PrePost 模块保留为源码选项，但这次不提供修复后 PrePost 实验的运行复现承诺。
- 源码按静态 import 闭包收集。原 Dense registry/models 的一些分支仍引用历史研究辅助模块，因此保留其兼容依赖；核心阅读入口以上表为准。Megatron Core 保留完整实际源码子树，避免替换为未经核实的 pip 版本。
- FLA 仅保留 AttnRes 所需依赖；两个 namespace 的 eager import 被移除，**算子文件未改**。来源及差异见 [PROVENANCE.md](docs/PROVENANCE.md)。
- 此次没有启动训练、改动历史运行或占用 GPU。整理版的验证范围以 [validation.json](provenance/validation.json) 为准；CPU/导入检查不等于完整 H200 训练验证。
- 结果表是已经完成的历史评测，不是本整理版重新训练得到的结果。无多种子显著性结论。

第三方代码保留原许可证及版权头，见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。本私有研究整理不另行授予第一方代码的公开使用许可。
