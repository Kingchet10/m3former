# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Research counters and audit hooks around the unchanged TorchTitan trainer."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from torchtitan.components.loss import IGNORE_INDEX, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from .artifacts import file_sha256, identity, read_manifest, write_json
from .models import (
    Architecture,
    local_training_operations,
    model_spec,
    parameter_report,
)
from .tracking import ResearchTracker


class ResearchMetrics(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.output = Path(kwargs["dump_folder"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.last_training = {}
        self.validation_callback = None

    def log(
        self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None
    ):
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        self.last_training = {
            "step": step,
            "loss": global_avg_loss,
            "grad_norm": grad_norm,
            **(extra_metrics or {}),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        super().log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics)

    def log_validation(self, loss, step, extra_metrics=None):
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss at step {step}")
        if torch.distributed.get_rank() == 0:
            with (self.output / "validation.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"step": step, "loss": loss, **(extra_metrics or {})})
                    + "\n"
                )
        super().log_validation(loss, step, extra_metrics)
        if self.validation_callback is not None:
            self.validation_callback(loss, step)


class ResearchTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        architecture: Architecture = field(default_factory=Architecture)
        stop_after_step: int | None = None
        token_budget: int | None = None
        audit_batches: bool = False
        tracking_dir: str | None = None
        resume_lineage: str | None = None
        verify_restored_checkpoint: bool = False
        paired_initialization: bool = False
        static_diagnostics_freq: int = 0

        def __post_init__(self):
            if self.paired_initialization:
                if self.debug.seed is None:
                    raise ValueError("Paired initialization requires an explicit seed")
                self.architecture.initialization_seed = self.debug.seed
            self.model_spec = model_spec(self.architecture)
            for _, loss, _, _ in self.loss.traverse(CrossEntropyLoss.Config):
                loss.global_vocab_size = self.architecture.vocab_size
            Trainer.Config.__post_init__(self)
            if self.token_budget is not None and self.token_budget <= 0:
                raise ValueError("token_budget must be positive")
            if self.metrics.log_freq != 1:
                raise ValueError("Research measurements require metrics.log_freq=1")
            if self.tracking_dir and not self.metrics.enable_tensorboard:
                raise ValueError("tracking_dir requires metrics.enable_tensorboard")
            if not 0 < self.training.seq_len <= self.architecture.max_seq_len:
                raise ValueError("Sequence length exceeds the configured RoPE range")
            if self.static_diagnostics_freq < 0:
                raise ValueError("Diagnostic frequency must be nonnegative")
            if self.architecture.ffn_kind != "dense" and (
                self.parallelism.tensor_parallel_degree != 1
                or self.parallelism.context_parallel_degree != 1
                or self.parallelism.pipeline_parallel_degree != 1
            ):
                raise ValueError(
                    "Static reference FFNs currently support data parallelism only"
                )

    def __init__(self, config):
        started = time.perf_counter()
        self.valid_tokens_seen = 0
        self.training_seconds_seen = 0.0
        self.nominal_flops_seen = 0
        self.tracker = None
        self.protocol_id = ""
        self.step_rows = []
        self.resume_lineage = None
        self.restored_checkpoint_verified = False
        self.dp_size = (
            config.parallelism.data_parallel_replicate_degree
            * config.parallelism.data_parallel_shard_degree
        )
        super().__init__(config)
        self.dp_size = self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard
        self.is_primary = torch.distributed.get_rank() == 0
        if (
            self.parallel_dims.world_size not in (1, 2, 4, 8, 16)
            or self.parallel_dims.world_size != self.dp_size
        ):
            raise ValueError(
                "Research execution supports only 1, 2, 4, 8, or 16 data-parallel ranks"
            )
        # Canonical global order and exact remaining-label masks are handled
        # by FrozenTokenLoader across each accumulation group. The parent
        # trainer normalizes all microbatches by the one global valid count.
        manifest = read_manifest(config.dataloader.dataset_path)
        experiment_files = {
            p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        }
        self.protocol_id = identity(
            {
                "artifact_id": manifest["artifact_id"],
                "architecture": asdict(config.architecture),
                "seq_len": config.training.seq_len,
                "local_batch": config.training.local_batch_size,
                "global_batch": config.training.global_batch_size,
                "precision": config.training.dtype,
                "mixed_precision": config.training.mixed_precision_param,
                "gradient_clip": config.training.max_norm,
                "token_budget": config.token_budget,
                "loss": config.loss.to_dict(),
                "compile": asdict(config.compile),
                "activation_checkpoint": None
                if config.activation_checkpoint is None
                else {
                    "type": type(config.activation_checkpoint).__qualname__,
                    "config": config.activation_checkpoint.to_dict(),
                },
                "parallelism": asdict(config.parallelism),
                "infinite_data": config.dataloader.infinite,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "optimizer": config.optimizer.to_dict(),
                "scheduler": config.lr_scheduler.to_dict(),
                "horizon_steps": config.training.steps,
                "seed": config.debug.seed,
                "deterministic": config.debug.deterministic,
                "experiment_files": experiment_files,
            }
        )
        self.output = Path(config.dump_folder)
        if config.resume_lineage:
            from .resume import validate_lineage

            self.resume_lineage = validate_lineage(config, manifest)
        elif config.verify_restored_checkpoint:
            raise ValueError(
                "A restored-checkpoint proof requires an explicit resume lineage"
            )
        self.model_report = parameter_report(
            config.architecture, self.model_parts[0], config.training.seq_len
        )
        if self.is_primary:
            write_json(self.output / "research-model.json", self.model_report)
        self.initialization_seconds = time.perf_counter() - started
        self.forward_retained_bytes = 0
        self._forward_entry_bytes = 0
        self.model_parts[0].register_forward_pre_hook(self._before_forward)
        self.model_parts[0].register_forward_hook(self._after_forward)
        self.manifest = manifest
        self.metrics_processor.validation_callback = self._track_validation

    def _ensure_tracker(self, completed_step):
        if (
            not self.is_primary
            or self.tracker is not None
            or not self.config.metrics.enable_tensorboard
        ):
            return
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[3],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        metadata = {
            "protocol_id": self.protocol_id,
            "artifact_id": self.manifest["artifact_id"],
            "tokenizer_sha256": self.manifest["tokenizer"]["tokenizer_sha256"],
            "seed": self.config.debug.seed,
            "git_commit": git,
            "configuration": self.config.to_dict(),
            "resume_lineage": self.resume_lineage,
            "time_axis": "cumulative measured training-step milliseconds; excludes validation, save, initialization and downtime",
        }
        self.tracker = ResearchTracker(
            self.config.tracking_dir or self.output / "research-tensorboard",
            metadata,
            step=completed_step,
            labels=self.valid_tokens_seen,
            training_seconds=self.training_seconds_seen,
            flops=self.nominal_flops_seen,
        )

    def _track_validation(self, loss, step):
        self._ensure_tracker(step)
        if self.tracker is not None:
            self.tracker.log(
                {
                    "validation/loss": loss,
                    "validation/perplexity": math.exp(loss)
                    if loss < 700
                    else float("inf"),
                },
                step=step,
                labels=self.valid_tokens_seen,
                training_seconds=self.training_seconds_seen,
                flops=self.nominal_flops_seen,
            )
            self.tracker.flush()

    def _before_forward(self, module, inputs):
        if module.training:
            self._forward_entry_bytes = torch.cuda.memory_allocated()

    def _after_forward(self, module, inputs, output):
        if module.training:
            self.forward_retained_bytes = max(
                self.forward_retained_bytes,
                torch.cuda.memory_allocated() - self._forward_entry_bytes,
            )

    def train_step(self, data_iterator):
        if (
            self.config.verify_restored_checkpoint
            and not self.restored_checkpoint_verified
        ):
            self._verify_restored_checkpoint()
        self._ensure_tracker(self.step - 1)
        for module in self.model_parts[0].modules():
            if hasattr(module, "set_schedule"):
                module.set_schedule(self.step - 1, self.config.training.steps)
        capture = self.config.static_diagnostics_freq > 0 and (
            self.step == 1 or self.step % self.config.static_diagnostics_freq == 0
        )
        for module in self.model_parts[0].modules():
            if hasattr(module, "capture_diagnostics"):
                module.capture_diagnostics = capture
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.forward_retained_bytes = 0
        valid = 0
        physical_tokens = 0
        attention_pairs = 0
        batch_hash = hashlib.sha256()
        remaining = (
            None
            if self.config.token_budget is None
            else self.config.token_budget - self.valid_tokens_seen
        )
        if self.dp_size > 1:
            self.dataloader.remaining_labels = remaining
            self.dataloader.audit_global_batch = self.config.audit_batches

        def audited():
            nonlocal valid, physical_tokens, attention_pairs
            while True:
                inputs, labels = next(data_iterator)
                physical_tokens += labels.numel()
                attention_pairs += int((inputs["positions"] + 1).clamp(max=self.config.architecture.attention_window or self.config.training.seq_len).sum())
                if remaining is not None:
                    available = max(remaining - valid, 0)
                    indices = (labels != IGNORE_INDEX).flatten().nonzero().flatten()
                    if len(indices) > available:
                        labels = labels.clone()
                        labels.flatten()[indices[available:]] = IGNORE_INDEX
                count = int((labels != IGNORE_INDEX).sum())
                valid += count
                if self.config.audit_batches:
                    for tensor in (inputs["input"], inputs["positions"], labels):
                        batch_hash.update(
                            tensor.numpy().astype("<i8", copy=False).tobytes()
                        )
                yield inputs, labels

        super().train_step(audited())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if self.dp_size > 1:
            counts = torch.tensor(
                [valid, physical_tokens, attention_pairs],
                dtype=torch.int64,
                device=self.device,
            )
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            valid, physical_tokens, attention_pairs = counts.tolist()
            measured = torch.tensor(
                [elapsed, self.forward_retained_bytes],
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(measured, op=torch.distributed.ReduceOp.MAX)
            elapsed, self.forward_retained_bytes = measured.tolist()
        self.valid_tokens_seen += valid
        self.training_seconds_seen += elapsed
        a = self.config.architecture
        attention_training_forward = (
            a.layers * 5 * a.heads * (a.dim // a.heads)**2 * physical_tokens
            if a.attention_kind == 'gla'
            else 4 * a.layers * a.dim * attention_pairs
        )
        row = {
            **self.metrics_processor.last_training,
            "step": self.step,
            "valid_labels": valid,
            "valid_labels_seen": self.valid_tokens_seen,
            "processed_input_tokens": physical_tokens,
            "nominal_training_flops": 3 * (
                self.model_report["linear_forward_flops_per_input_token"]
                * physical_tokens + attention_training_forward
            ),
            "training_seconds": elapsed,
            "valid_labels_per_second": valid / elapsed,
            "forward_retained_increment_bytes": self.forward_retained_bytes,
        }
        local_operations = local_training_operations(
            self.config.architecture,
            physical_tokens,
            microbatches=self.dp_size * self.gradient_accumulation_steps,
        )
        row["local_training_flops"] = local_operations
        if self.config.architecture.ffn_kind == "blast":
            from .structured_ops import coupling_training_flops

            a = self.config.architecture
            # Informational breakdown; already included in linear GEMM FLOPs.
            row["projection_coupling_training_flops"] = (
                a.layers
                * coupling_training_flops(
                    a.ffn_kind, a.projection_blocks, a.projection_rank, physical_tokens
                )
            )
        row["nominal_training_flops"] += local_operations
        # Common activation accounting is reported independently of MACs.
        # SiLU uses one exp and one reciprocal per hidden feature in forward;
        # its direct forward+VJP and product use 11 basic arithmetic operations.
        row["swiglu_basic_training_ops"] = (
            (11 if self.config.architecture.activation == "swiglu" else 13)
            * physical_tokens
            * self.config.architecture.ff_dim
            * self.config.architecture.layers
        )
        row["swiglu_exp_reciprocal_ops"] = (
            2
            * physical_tokens
            * self.config.architecture.ff_dim
            * self.config.architecture.layers
        )
        row["nominal_training_flops"] += row["swiglu_basic_training_ops"]
        if self.config.architecture.activation == "gelu":
            row["gelu_basic_training_ops"] = row.pop("swiglu_basic_training_ops")
            row["gelu_erf_exp_ops"] = row.pop("swiglu_exp_reciprocal_ops")
        if capture:
            for name, module in self.model_parts[0].named_modules():
                coefficients = getattr(module, "coefficients", None)
                if coefficients is not None:
                    row[f"static/{name}/coefficient_norm_after_update"] = float(
                        coefficients.detach().float().norm()
                    )
                    if coefficients.grad is None:
                        raise RuntimeError(f"Missing quadratic gradient: {name}")
                    row[f"static/{name}/coefficient_grad_norm_after_clip"] = float(
                        coefficients.grad.detach().float().norm()
                    )
                if getattr(module, "a", None) is not None and module.a.requires_grad:
                    if module.a.grad is None:
                        raise RuntimeError(f"Missing S2 A gradient: {name}")
                    row[f"static/{name}/a_grad_norm_after_clip"] = float(
                        module.a.grad.detach().float().norm()
                    )
                for key, value in getattr(module, "diagnostics", {}).items():
                    number = float(value)
                    if not math.isfinite(number):
                        raise RuntimeError(
                            f"Non-finite static diagnostic: {name}/{key}"
                        )
                    if (getattr(module, "kind", None) == "e_ref"
                        and key == "max_vector_relative_residual" and number > 2e-5):
                        raise RuntimeError(f"E-Ref stationarity tolerance exceeded: {name}: {number}")
                    row[f"static/{name}/{key}"] = number
                if hasattr(module, "levels") and hasattr(module, "active_depth"):
                    for i, parameter in enumerate(module.levels):
                        if i < module.active_depth:
                            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                                raise RuntimeError(f"Invalid CF gradient: {name}/depth{i+1}")
                            row[f"static/{name}/depth{i+1}_grad_norm"] = float(parameter.grad.norm())
                        elif parameter.grad is not None:
                            raise RuntimeError(f"Frozen CF parameter has a gradient: {name}/depth{i+1}")
                for parameter_name in ("d_raw", "local_w1", "local_w2", "local_theta"):
                    parameter = getattr(module, parameter_name, None)
                    if parameter is not None:
                        if parameter.grad is None:
                            raise RuntimeError(
                                f"Missing local gradient: {name}/{parameter_name}"
                            )
                        row[f"static/{name}/{parameter_name}_grad_norm_after_clip"] = (
                            float(parameter.grad.detach().float().norm())
                        )
                if hasattr(module, "kind") and module.kind in (
                    "low_rank",
                    "monarch",
                    "blast",
                ):
                    for factor_name, parameter in module.named_parameters(
                        recurse=False
                    ):
                        if parameter.grad is None:
                            raise RuntimeError(
                                f"Missing projection gradient: {name}/{factor_name}"
                            )
                        row[f"static/{name}/{factor_name}_grad_norm_after_clip"] = (
                            float(parameter.grad.detach().float().norm())
                        )
            if not all(
                math.isfinite(v) for k, v in row.items() if k.startswith("static/")
            ):
                raise RuntimeError("Non-finite static coefficient or gradient")
        if self.config.audit_batches:
            row["batch_sha256"] = (
                self.dataloader.last_global_batch_sha256
                if self.dp_size > 1
                else batch_hash.hexdigest()
            )
        self.nominal_flops_seen += row["nominal_training_flops"]
        row["training_seconds_seen"] = self.training_seconds_seen
        row["nominal_training_flops_seen"] = self.nominal_flops_seen
        if self.tracker is not None:
            self.tracker.log(
                {
                    "train/loss": row["loss"],
                    "train/grad_norm": row["grad_norm"],
                    "train/valid_labels_per_second": row["valid_labels_per_second"],
                    "progress/valid_labels": self.valid_tokens_seen,
                    "progress/training_seconds": self.training_seconds_seen,
                    "progress/nominal_training_flops": self.nominal_flops_seen,
                    "memory/max_allocated_bytes": row["max_allocated_bytes"],
                    "memory/max_reserved_bytes": row["max_reserved_bytes"],
                    **{k: v for k, v in row.items() if k.startswith("lr")},
                    **{k: v for k, v in row.items() if k.startswith("static/")},
                },
                step=self.step,
                labels=self.valid_tokens_seen,
                training_seconds=self.training_seconds_seen,
                flops=self.nominal_flops_seen,
            )
        self.step_rows.append(row)
        if self.is_primary:
            with (self.output / "research-steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    def should_continue_training(self):
        return (
            super().should_continue_training()
            and (
                self.config.stop_after_step is None
                or self.step < self.config.stop_after_step
            )
            and (
                self.config.token_budget is None
                or self.valid_tokens_seen < self.config.token_budget
            )
        )

    def state_dict(self):
        return {
            **super().state_dict(),
            "ntokens_seen": self.ntokens_seen * self.dp_size,
            "research_valid_tokens": self.valid_tokens_seen,
            "research_protocol_id": self.protocol_id,
            "research_training_seconds": self.training_seconds_seen,
            "research_nominal_flops": self.nominal_flops_seen,
        }

    def load_state_dict(self, state):
        if state["research_protocol_id"] != self.protocol_id:
            if (
                self.resume_lineage is None
                or state["research_protocol_id"]
                != self.resume_lineage["parent_protocol_id"]
            ):
                raise ValueError("Research protocol changed across checkpoint resume")
            expected = self.resume_lineage["checkpoint_state"]
            if any(
                state[key.removeprefix("train_state.")] != value
                for key, value in expected.items()
            ):
                raise ValueError(
                    "Loaded research counters differ from the checkpoint lineage"
                )
        super().load_state_dict(state)
        if self.ntokens_seen % self.dp_size:
            raise ValueError("Processed token counter does not divide across DP ranks")
        self.ntokens_seen //= self.dp_size
        self.valid_tokens_seen = state["research_valid_tokens"]
        self.training_seconds_seen = state["research_training_seconds"]
        self.nominal_flops_seen = state["research_nominal_flops"]

    def _verify_restored_checkpoint(self):
        restored_step = self.resume_lineage["checkpoint_state"]["train_state.step"]
        if self.step != restored_step + 1:
            raise ValueError(
                "Restore proof must run before the first resumed optimizer step"
            )
        self.step = restored_step
        try:
            self.checkpointer.save(restored_step, last_step=True)
        finally:
            self.step = restored_step + 1
        proof_path = self.output / "restored-checkpoint-verification.json"
        status = torch.zeros((), dtype=torch.int64, device=self.device)
        if self.is_primary:
            command = [
                sys.executable,
                "-m",
                "torchtitan.experiments.brainfuse_scaling.resume",
                "verify",
                "--parent",
                self.resume_lineage["parent_checkpoint"],
                "--restored",
                str(self.output / "checkpoint" / f"step-{restored_step}"),
                "--output",
                str(proof_path),
            ]
            with (self.output / "restored-checkpoint-verification.log").open(
                "x"
            ) as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                    timeout=180,
                )
            status.fill_(result.returncode == 0)
        torch.distributed.broadcast(status, src=0)
        if not bool(status.item()):
            raise ValueError(
                "Full checkpoint restore failed independent tensor/state comparison"
            )
        self.restored_checkpoint_verified = True

    def train(self):
        started = time.perf_counter()
        status = "failed"
        try:
            from .cofr_operator import CF_KINDS, prepare_adam_state
            if self.config.architecture.ffn_kind in CF_KINDS:
                for optimizer in self.optimizers:
                    prepare_adam_state(optimizer)
            super().train()
            if self.step_rows and self.step_rows[-1]["step"] != self.step:
                raise RuntimeError(
                    "Frozen data ended during an incomplete optimizer step"
                )
            if (
                self.config.checkpoint.enable
                and self.step != self.config.training.steps
            ):
                self.checkpointer.save(self.step, last_step=True)
            if (
                self.config.token_budget is not None
                and self.valid_tokens_seen < self.config.token_budget
                and self.config.stop_after_step is None
            ):
                raise RuntimeError(
                    "Run ended before the requested effective token budget"
                )
            if (
                self.config.validator.enable
                and self.step_rows
                and not self.validator.should_validate(self.step)
            ):
                # The upstream validator runs at step 1 and fixed intervals.
                # Exact label budgets usually end between those intervals.
                self.validator.validate(self.model_parts, self.step)
            status = "complete"
        finally:
            if self.tracker is not None:
                self.tracker.flush()
            elapsed = time.perf_counter() - started
            labels = sum(row["valid_labels"] for row in self.step_rows)
            if self.is_primary:
                self._write_summary(status, labels, elapsed)

    def _write_summary(self, status, labels, elapsed):
        write_json(
            self.output / "research-summary.json",
            {
                "status": status,
                "protocol_id": self.protocol_id,
                "step": self.step,
                "valid_labels_seen": self.valid_tokens_seen,
                "training_seconds_seen": self.training_seconds_seen,
                "nominal_training_flops_seen": self.nominal_flops_seen,
                "this_process_valid_labels": labels,
                "initialization_seconds": self.initialization_seconds,
                "train_loop_seconds": elapsed,
                "end_to_end_valid_labels_per_second": labels
                / (elapsed + self.initialization_seconds),
                "train_loop_valid_labels_per_second": labels / elapsed,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "gpu_count": self.dp_size,
                "model": self.model_report,
            },
        )

    def close(self):
        if self.tracker is not None:
            self.tracker.close()
        super().close()
