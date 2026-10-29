# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python recipes; CLI overrides preserve the common research protocol."""

import math
import os
from pathlib import Path

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)

from .loader import FrozenTokenLoader
from .models import architecture
from .trainer import ResearchMetrics, ResearchTrainer
from .validation import ResearchValidator


def recipe(flavor):
    root = Path(os.environ.get("BRAINFUSE_DATA_ROOT", "./data/brainfuse-fineweb"))
    a = architecture(flavor)
    return ResearchTrainer.Config(
        architecture=a,
        hf_assets_path=str(root / "tokenizer"),
        dataloader=FrozenTokenLoader.Config(dataset_path=str(root), pin_memory=True),
        # Full CE avoids the chunked backward/FSDP failure on our pinned runtime.
        # Its extra activation memory is included in the throughput sweep.
        loss=CrossEntropyLoss.Config(global_vocab_size=a.vocab_size),
        optimizer=default_adamw(lr=1e-3),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20, decay_ratio=0.8, decay_type="cosine", min_lr_factor=0.1
        ),
        training=TrainingConfig(
            local_batch_size=4,
            global_batch_size=16,
            seq_len=2048,
            steps=200,
            dtype="float32",
            mixed_precision_param="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=True, interval=100, last_save_model_only=False
        ),
        metrics=ResearchMetrics.Config(log_freq=1, disable_color_printing=True),
        validator=ResearchValidator.Config(
            enable=True,
            freq=100,
            steps=8,
            dataloader=FrozenTokenLoader.Config(
                dataset_path=str(root), split="validation", pin_memory=True
            ),
        ),
        debug=DebugConfig(seed=42, save_config_file="config.json"),
    )


def dense_30m():
    return recipe("30m")


def dense_100m():
    return recipe("100m")


def dense_10m():
    return recipe("10m")


def formal_recipe(flavor):
    """Locked first-matrix schedule; LR and seed are explicit run overrides."""
    config = recipe(flavor)
    config.token_budget = {
        "10m": 500_000_000,
        "30m": 1_000_000_000,
        "100m": 3_000_000_000,
    }[flavor]
    config.training.local_batch_size = 16
    config.training.steps = math.ceil(config.token_budget / (16 * 2048))
    config.lr_scheduler.warmup_steps = math.ceil(config.training.steps / 100)
    config.lr_scheduler.decay_ratio = None
    config.checkpoint.interval = math.ceil(100_000_000 / (16 * 2048))
    # Preserve every first-matrix checkpoint, including the best validation step.
    # Any later cleanup must first record and retain the chosen checkpoints.
    config.checkpoint.keep_latest_k = 0
    config.validator.freq = config.checkpoint.interval
    config.validator.steps = -1
    config.metrics.enable_tensorboard = True
    return config


def formal_dense_30m():
    return formal_recipe("30m")


def formal_dense_100m():
    return formal_recipe("100m")


def formal_dense_10m():
    return formal_recipe("10m")


def s_milestone_recipe(kind, hidden=1024):
    config = formal_recipe("30m")
    config.architecture.ffn_kind = kind
    config.architecture.ff_dim = hidden
    config.architecture.initialization_reference_width = 1024
    config.paired_initialization = True
    config.debug.deterministic = True
    config.static_diagnostics_freq = 100
    config.audit_batches = True
    return config


def s_milestone_b0():
    return s_milestone_recipe("dense")


def s_milestone_b1():
    return s_milestone_recipe("dense", hidden=853)


def s_milestone_s0():
    return s_milestone_recipe("s0")


def s_milestone_s1_d():
    return s_milestone_recipe("s1_d")


def s_milestone_s1_q():
    return s_milestone_recipe("s1_q")


def w_milestone_b2():
    return s_milestone_recipe("low_rank")


def w_milestone_w1_m():
    return s_milestone_recipe("monarch")


def w_milestone_w1_b():
    config = s_milestone_recipe("blast")
    config.architecture.projection_rank = 138
    return config


def s2_milestone_recipe(kind):
    config = s_milestone_recipe(kind, hidden=928 if kind == "s2_mid" else 896)
    config.architecture.anchor_width = 480 if kind == "s2_mid" else 640
    return config


def s2_milestone_s2_l():
    return s2_milestone_recipe("s2_l")


def s2_milestone_s2_q():
    return s2_milestone_recipe("s2_q")


def s2_milestone_s2_blocked():
    return s2_milestone_recipe("s2_blocked")


def s2_milestone_s2_mid():
    return s2_milestone_recipe("s2_mid")


def s3_milestone_recipe(kind):
    from .s3_linear import RECIPES

    anchor, dendritic, k, mode = RECIPES[kind]
    config = s_milestone_recipe(kind, hidden=anchor + dendritic)
    config.architecture.anchor_width = anchor
    config.architecture.local_outputs = k
    config.architecture.local_mode = mode
    return config


def s3_milestone_s3_light():
    return s3_milestone_recipe("s3_light")


def s3_milestone_s3_mid():
    return s3_milestone_recipe("s3_mid")


def s3_milestone_s3_25ga():
    return s3_milestone_recipe("s3_25ga")


def s3_milestone_s3_midga():
    return s3_milestone_recipe("s3_midga")


def s3_milestone_s3_k3():
    return s3_milestone_recipe("s3_k3")


def s3_milestone_s3_k3ga():
    return s3_milestone_recipe("s3_k3ga")


def frontier_milestone_recipe(kind):
    from .frontier_linear import RECIPES

    ha, hd, b, k, mode = RECIPES[kind]
    config = s_milestone_recipe(kind, hidden=ha + hd)
    config.architecture.anchor_width = ha
    config.architecture.local_group_size = b
    config.architecture.local_outputs = k
    config.architecture.local_mode = mode
    return config


def frontier_milestone_frontier_n10():
    return frontier_milestone_recipe("frontier_n10")


def frontier_milestone_frontier_n15():
    return frontier_milestone_recipe("frontier_n15")


def frontier_milestone_frontier_n20():
    return frontier_milestone_recipe("frontier_n20")


def frontier_milestone_frontier_n25():
    return frontier_milestone_recipe("frontier_n25")


def frontier_milestone_frontier_n30():
    return frontier_milestone_recipe("frontier_n30")


def frontier_milestone_frontier_b3k2():
    return frontier_milestone_recipe("frontier_b3k2")


def frontier_milestone_frontier_b5k4():
    return frontier_milestone_recipe("frontier_b5k4")


def frontier_milestone_frontier_b6k5():
    return frontier_milestone_recipe("frontier_b6k5")


def frontier_milestone_frontier_ortho():
    return frontier_milestone_recipe("frontier_ortho")


def frontier_milestone_frontier_fixed():
    return frontier_milestone_recipe("frontier_fixed")


def rrp_milestone_recipe(kind):
    from .rrp_linear import RECIPES

    ha, hd, b, k, mode = RECIPES[kind]
    config = s_milestone_recipe(kind, hidden=ha + hd)
    config.architecture.anchor_width = ha
    config.architecture.local_group_size = b
    config.architecture.local_outputs = k
    config.architecture.local_mode = mode
    return config


def rrp_milestone_rrp1():
    return rrp_milestone_recipe("rrp1")


def rrp_milestone_rrp2():
    return rrp_milestone_recipe("rrp2")


def rrp_milestone_rrp3():
    return rrp_milestone_recipe("rrp3")


def rrp_milestone_rrp2_flat():
    return rrp_milestone_recipe("rrp2_flat")


def rrp_milestone_rrp2_learn():
    return rrp_milestone_recipe("rrp2_learn")


def rrp_speed_a1():
    from .rrp_speed import speed_recipe

    return speed_recipe("a1")


def rrp_speed_a2():
    from .rrp_speed import speed_recipe

    return speed_recipe("a2")


def rrp_speed_dense768():
    from .rrp_speed import speed_recipe

    return speed_recipe("dense768")


def rrp_speed_dense800():
    from .rrp_speed import speed_recipe

    return speed_recipe("dense800")


def rrp_speed_dense832():
    from .rrp_speed import speed_recipe

    return speed_recipe("dense832")


def e_recipe(name, flavor="30m", seed=47):
    from .e_operator import MODELS
    if name not in MODELS:
        raise ValueError(name)
    kind = "dense" if name == "dense" else "p_mlp" if name == "mlp" else "e_" + name
    config = formal_recipe(flavor)
    config.architecture.ffn_kind = kind
    config.architecture.initialization_reference_width = config.architecture.ff_dim
    config.paired_initialization = True
    config.debug.deterministic = True
    config.static_diagnostics_freq = 100
    config.audit_batches = True
    config.training.local_batch_size = 4
    config.debug.seed = seed
    config.optimizer.param_groups[0].optimizer_kwargs["lr"] = 0.003
    config.architecture.operator_alpha = 1.0
    config.architecture.operator_compile = True
    return config


def e_g0_dense():
    return e_recipe("dense")


def e_g0_mlp():
    return e_recipe("mlp")


def e_g0_quadratic():
    return e_recipe("quadratic")


def e_g0_closed():
    return e_recipe("closed")


def e_g0_k1():
    return e_recipe("k1")


def e_g0_k4():
    return e_recipe("k4")


def e_g0_ref():
    return e_recipe("ref")


def e_rep_30m_dense():
    return e_recipe("dense", "30m", 48)


def e_rep_30m_mlp():
    return e_recipe("mlp", "30m", 48)


def e_rep_30m_k4():
    return e_recipe("k4", "30m", 48)


def e_rep_30m_quadratic():
    return e_recipe("quadratic", "30m", 48)


def e_rep_10m_dense():
    return e_recipe("dense", "10m", 47)


def e_rep_10m_mlp():
    return e_recipe("mlp", "10m", 47)


def e_rep_10m_k4():
    return e_recipe("k4", "10m", 47)


def g1_recipe(name):
    if name in ("mlp", "quadratic", "k4"):
        config = e_recipe(name)
        config.architecture.output_compression = 2
    elif name == "narrow":
        config = s_milestone_recipe("dense", hidden=853)
    elif name in ("s0", "s1q"):
        config = s_milestone_recipe("s0" if name == "s0" else "s1_q")
    elif name == "rrp":
        config = rrp_milestone_recipe("rrp2_learn")
    else:
        raise ValueError(name)
    config.training.local_batch_size = 4
    config.debug.seed = 47
    config.optimizer.param_groups[0].optimizer_kwargs["lr"] = 0.003
    return config


def g1_narrow():
    return g1_recipe("narrow")


def cofr_recipe(name):
    from .cofr_operator import MODELS
    if name not in MODELS:
        raise ValueError(name)
    config = g1_recipe("s0")
    config.architecture.ffn_kind = "local_safe_cf" if name.startswith("safe") else "local_cf"
    config.architecture.output_compression = 2
    config.architecture.cf_depth = 1 if "d1" in name else 3
    config.architecture.cf_schedule = name.endswith("schedule")
    # More frequent resumable checkpoints; validation/data/optimizer unchanged.
    config.checkpoint.interval = 512
    config.checkpoint.keep_latest_k = 3
    return config


def lf_recipe(name):
    from .local_functions import MODELS
    if name not in MODELS:
        raise ValueError(name)
    config = s_milestone_recipe("dense" if name in ("dense","narrow") else "lf_"+name,
                                hidden=853 if name == "narrow" else 1024)
    config.training.local_batch_size = 4
    config.debug.seed = 47
    config.optimizer.param_groups[0].optimizer_kwargs["lr"] = 0.003
    if name not in ("dense","narrow"):
        config.architecture.output_compression = 2
        config.architecture.local_group_size = 2
        config.architecture.local_outputs = 1
    config.checkpoint.interval = 512
    config.checkpoint.keep_latest_k = 3
    return config


def lf_dense():
    return lf_recipe("dense")

def lf_narrow():
    return lf_recipe("narrow")

def lf_sum():
    return lf_recipe("sum")

def lf_normalized_linear():
    return lf_recipe("normalized_linear")

def lf_diagonal_quadratic():
    return lf_recipe("diagonal_quadratic")

def lf_full_quadratic():
    return lf_recipe("full_quadratic")

def lf_local_mlp():
    return lf_recipe("local_mlp")

def lf_pi_sigma2():
    return lf_recipe("pi_sigma2")

def lf_pi_sigma4():
    return lf_recipe("pi_sigma4")

def lf_chebyshev():
    return lf_recipe("chebyshev")

def lf_safe_rational():
    return lf_recipe("safe_rational")


def safe_rational_4to2():
    config=lf_recipe('safe_rational')
    config.architecture.ffn_kind='lf_safe_rational_4to2'
    config.architecture.local_group_size=4
    config.architecture.local_outputs=2
    return config


def safe_rational_100m():
    """Pure fixed-pair B2 Safe-Rational on the established 100M/3B protocol."""
    config = formal_recipe("100m")
    a = config.architecture
    a.ffn_kind = "lf_safe_rational"
    a.initialization_reference_width = a.ff_dim
    a.output_compression = 2
    a.local_group_size = 2
    a.local_outputs = 1
    a.dpr_shifts = a.ridge_rank = 0
    config.paired_initialization = True
    config.debug.deterministic = True
    config.debug.seed = 47
    config.audit_batches = True
    config.static_diagnostics_freq = 100
    config.training.local_batch_size = 8
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.spmd_backend = "partial_dtensor"
    config.optimizer.param_groups[0].optimizer_kwargs["lr"] = .001
    config.checkpoint.interval = 512
    config.checkpoint.keep_latest_k = 3
    return config

def dpr1():
    config = lf_recipe('safe_rational')
    config.architecture.dpr_shifts = 1
    return config

def rrm_recipe(rank,kind='rational'):
    config=lf_recipe('safe_rational')
    config.architecture.ridge_rank=rank
    config.architecture.ridge_kind=kind
    return config

def rrm4(): return rrm_recipe(4)
def rrm8(): return rrm_recipe(8)
def ridge_linear8(): return rrm_recipe(8,'linear')
def ridge_silu8(): return rrm_recipe(8,'silu')

def dpr4():
    config = lf_recipe('safe_rational')
    config.architecture.dpr_shifts = 4
    return config

def lf_rbf():
    return lf_recipe("rbf")

def lf_cmac():
    return lf_recipe("cmac")

def lf_cmac_mlp():
    return lf_recipe("cmac_mlp")


def cofr_cf_d1():
    return cofr_recipe("cf_d1")


def cofr_cf_d3():
    return cofr_recipe("cf_d3")


def cofr_cf_d3_schedule():
    return cofr_recipe("cf_d3_schedule")


def cofr_safe_d1():
    return cofr_recipe("safe_d1")


def cofr_safe_d3():
    return cofr_recipe("safe_d3")


def cofr_safe_d3_schedule():
    return cofr_recipe("safe_d3_schedule")


def g1_s0():
    return g1_recipe("s0")


def g1_mlp():
    return g1_recipe("mlp")


def g1_quadratic():
    return g1_recipe("quadratic")


def g1_k4():
    return g1_recipe("k4")


def g1_s1q():
    return g1_recipe("s1q")


def g1_rrp():
    return g1_recipe("rrp")


def p_recipe(name):
    from .p_operator import TRAIN_MODELS

    if name not in TRAIN_MODELS:
        raise ValueError("Unknown model in the frozen P/G0 matrix")
    config = s_milestone_recipe("dense" if name == "p_dense" else name)
    config.training.local_batch_size = 4
    config.architecture.operator_alpha = 1.0
    config.architecture.operator_compile = True
    return config


def p_g0_dense():
    return p_recipe("p_dense")


def p_g0_mlp():
    return p_recipe("p_mlp")


def p_g0_scale():
    return p_recipe("p_scale")


def p_g0_diag():
    return p_recipe("p_diag")


def p_g0_direct():
    return p_recipe("p_direct")


def p_g0_k1():
    return p_recipe("p_k1")


def p_g0_k4():
    return p_recipe("p_k4")


def rrp_scaling_10m_b0():
    from .rrp_scaling import scaling_recipe

    return scaling_recipe("10m", "b0")


def rrp_scaling_10m_b1():
    from .rrp_scaling import scaling_recipe

    return scaling_recipe("10m", "b1")


def rrp_scaling_10m_k3():
    from .rrp_scaling import scaling_recipe

    return scaling_recipe("10m", "k3")


def rrp_scaling_10m_rrp2_learn():
    from .rrp_scaling import scaling_recipe

    return scaling_recipe("10m", "rrp2_learn")


def rrp_scaling_100m_b0():
    from .rrp_scaling import scaling_recipe

    return scaling_recipe("100m", "b0")


def rrp_scaling_100m_b1():
    from .rrp_scaling import scaling_recipe

    return scaling_recipe("100m", "b1")


def rrp_scaling_100m_k3():
    from .rrp_scaling import scaling_recipe

    return scaling_recipe("100m", "k3")


def rrp_scaling_100m_rrp2_learn():
    from .rrp_scaling import scaling_recipe

    return scaling_recipe("100m", "rrp2_learn")


# Explicit names remain discoverable by ConfigManager; widths live in ridge_grid.
def grid_b2_r0():
    from .ridge_grid import recipe
    return recipe('grid_b2_r0')
def grid_b2_r4():
    from .ridge_grid import recipe
    return recipe('grid_b2_r4')
def grid_b2_r8():
    from .ridge_grid import recipe
    return recipe('grid_b2_r8')
def grid_b2_r16():
    from .ridge_grid import recipe
    return recipe('grid_b2_r16')
def grid_b4_r0():
    from .ridge_grid import recipe
    return recipe('grid_b4_r0')
def grid_b4_r4():
    from .ridge_grid import recipe
    return recipe('grid_b4_r4')
def grid_b4_r8():
    from .ridge_grid import recipe
    return recipe('grid_b4_r8')
def grid_b4_r16():
    from .ridge_grid import recipe
    return recipe('grid_b4_r16')
def grid_b8_r0():
    from .ridge_grid import recipe
    return recipe('grid_b8_r0')
def grid_b8_r4():
    from .ridge_grid import recipe
    return recipe('grid_b8_r4')
def grid_b8_r8():
    from .ridge_grid import recipe
    return recipe('grid_b8_r8')
def grid_b8_r16():
    from .ridge_grid import recipe
    return recipe('grid_b8_r16')
def grid_b2_silu8():
    from .ridge_grid import recipe
    return recipe('grid_b2_silu8')
def grid_b4_silu8():
    from .ridge_grid import recipe
    return recipe('grid_b4_silu8')
def grid_b8_silu8():
    from .ridge_grid import recipe
    return recipe('grid_b8_silu8')


def neuron_complexity_step1():
    from .neuron_protocol import recipe
    return recipe()


def neuron_complexity_step2():
    from .neuron_protocol_step2 import recipe
    return recipe()


def stagea():
    from .stagea_protocol import recipe
    return recipe()


def wsd():
    from .wsd_runtime import wsd_recipe
    return wsd_recipe()
