# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
import math
import os
from typing import ClassVar, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    ProcessGroupCollection,
    apply_random_logits,
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
    get_tokens_per_expert_and_token_count,
    router_gating_linear,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
    z_loss_func,
)
from megatron.core.transformer.moe.router_replay import RouterReplay
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.tp_cp_group = pg_collection.tp_cp
        self.tp_dp_cp_group = pg_collection.tp_dp_cp

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if self.config.add_bias_linear:
            self.bias = torch.nn.Parameter(
                torch.empty((self.config.num_moe_experts), dtype=torch.float32)
            )
        else:
            self.bias = None
        # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
        # So we need to know if the model is configured to calculate per token loss.
        self.calculate_per_token_loss = self.config.calculate_per_token_loss
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the router parameters."""
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
            if self.bias is not None:
                self.config.init_method(self.bias)
        self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype=self.config.params_dtype)
            setattr(self.bias, 'sequence_parallel', self.config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class _GeoPathBlock(torch.nn.Module):
    """Block-shared GeoPath dynamics module."""

    def __init__(self, config: TransformerConfig, num_cells: int) -> None:
        super().__init__()
        self.config = config
        self.num_cells = num_cells
        self.grid_h, self.grid_w = self._infer_grid_shape(num_cells)
        self.fourier_bands = 4
        self.fourier_dim = 2 * 2 * self.fourier_bands
        path_hidden_size = min(max(self.config.hidden_size // 2, 64), 256)
        mlp_input_dim = self.config.hidden_size + self.fourier_dim + 2

        self.velocity_update = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_dim, path_hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(path_hidden_size, 2),
        )
        self.velocity_gate = torch.nn.Linear(mlp_input_dim, 2, bias=True)
        self.position_reset = torch.nn.Linear(self.config.hidden_size, 2, bias=True)
        self.position_reset_gate = torch.nn.Linear(self.config.hidden_size, 1, bias=True)

        frequencies = (2.0 ** torch.arange(self.fourier_bands, dtype=torch.float32)) * (
            2.0 * math.pi
        )
        self.register_buffer('fourier_frequencies', frequencies, persistent=False)
        self.register_buffer(
            'cell_positions',
            self._build_cell_positions(self.grid_h, self.grid_w),
            persistent=False,
        )
        self.reset_parameters()

    @staticmethod
    def _infer_grid_shape(num_cells: int) -> tuple[int, int]:
        grid_h = int(math.sqrt(num_cells))
        while grid_h > 1 and num_cells % grid_h != 0:
            grid_h -= 1
        grid_w = num_cells // grid_h
        return grid_h, grid_w

    @staticmethod
    def _build_cell_positions(grid_h: int, grid_w: int) -> torch.Tensor:
        positions = []
        for i in range(grid_h):
            for j in range(grid_w):
                positions.append(((i + 0.5) / grid_h, (j + 0.5) / grid_w))
        return torch.tensor(positions, dtype=torch.float32)

    def reset_parameters(self) -> None:
        if self.config.perform_initialization:
            self.config.init_method(self.velocity_update[0].weight)
            torch.nn.init.zeros_(self.velocity_update[0].bias)
            self.config.output_layer_init_method(self.velocity_update[2].weight)
            torch.nn.init.zeros_(self.velocity_update[2].bias)
            self.config.init_method(self.velocity_gate.weight)
            torch.nn.init.zeros_(self.velocity_gate.bias)
            self.config.init_method(self.position_reset.weight)
            torch.nn.init.zeros_(self.position_reset.bias)
            self.config.init_method(self.position_reset_gate.weight)
            torch.nn.init.zeros_(self.position_reset_gate.bias)

        for linear in (
            self.velocity_update[0],
            self.velocity_update[2],
            self.velocity_gate,
            self.position_reset,
            self.position_reset_gate,
        ):
            linear.weight.data = linear.weight.data.to(dtype=self.config.params_dtype)
            linear.bias.data = linear.bias.data.to(dtype=self.config.params_dtype)
            setattr(linear.weight, 'sequence_parallel', self.config.sequence_parallel)
            setattr(linear.bias, 'sequence_parallel', self.config.sequence_parallel)


class GeoPathRouter(MegatronModule):
    """GeoPath router: block-shared path dynamics plus fused per-layer content logits."""

    supports_routing_state = True
    _shared_path_blocks: ClassVar[dict[tuple[int, int], _GeoPathBlock]] = {}

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.topk = self.config.moe_router_topk
        self.eta = float(self.config.stateful_router_eta)
        self.num_cells = int(self.config.geopath_router_num_cells)
        self.block_size = int(self.config.geopath_router_block_size)
        self.prior_weight = float(self.config.geopath_router_prior_weight)
        self.cell_temperature = float(self.config.geopath_router_cell_temperature)
        self.balance_coeff = float(self.config.geopath_router_balance_coeff)
        self.velocity_coeff = float(self.config.geopath_router_velocity_coeff)
        self.acceleration_coeff = float(self.config.geopath_router_acceleration_coeff)
        self.turn_coeff = float(self.config.geopath_router_turn_coeff)
        self.layer_number = None
        self.tp_cp_group = pg_collection.tp_cp
        self.calculate_per_token_loss = self.config.calculate_per_token_loss

        self.experts_per_cell = self.num_experts // self.num_cells
        expert_to_cell = torch.arange(self.num_experts, dtype=torch.long) // self.experts_per_cell
        self.register_buffer('expert_to_cell', expert_to_cell, persistent=False)

        self.content_weight = torch.nn.Parameter(
            torch.empty((self.num_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if self.config.add_bias_linear:
            self.content_bias = torch.nn.Parameter(
                torch.empty((self.num_experts), dtype=torch.float32)
            )
        else:
            self.content_bias = None
        self.state_init = torch.nn.Linear(self.config.hidden_size, 2, bias=True)

        self.path_block: Optional[_GeoPathBlock] = None
        self._is_first_moe_layer = False
        self._is_block_start = False
        self._incoming_state: Optional[torch.Tensor] = None
        self._outgoing_state: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.config.perform_initialization:
            self.config.init_method(self.content_weight)
            if self.content_bias is not None:
                self.config.init_method(self.content_bias)
            self.config.init_method(self.state_init.weight)
            torch.nn.init.zeros_(self.state_init.bias)
        self.content_weight.data = self.content_weight.data.to(dtype=self.config.params_dtype)
        setattr(self.content_weight, 'sequence_parallel', self.config.sequence_parallel)
        if self.content_bias is not None:
            self.content_bias.data = self.content_bias.data.to(dtype=self.config.params_dtype)
            setattr(self.content_bias, 'sequence_parallel', self.config.sequence_parallel)
        self.state_init.weight.data = self.state_init.weight.data.to(dtype=self.config.params_dtype)
        self.state_init.bias.data = self.state_init.bias.data.to(dtype=self.config.params_dtype)
        setattr(self.state_init.weight, 'sequence_parallel', self.config.sequence_parallel)
        setattr(self.state_init.bias, 'sequence_parallel', self.config.sequence_parallel)

    def _ensure_path_block(self) -> None:
        if self.path_block is not None:
            return
        key = (id(self.config), 0)
        shared_block = self._shared_path_blocks.get(key)
        if shared_block is None:
            shared_block = _GeoPathBlock(self.config, self.num_cells)
            self._shared_path_blocks[key] = shared_block
        self.path_block = shared_block

    def _get_moe_ordinal(self, layer_number: int) -> int:
        moe_layer_freq = self.config.moe_layer_freq
        if isinstance(moe_layer_freq, list):
            active_layers = moe_layer_freq[:layer_number]
            return sum(bool(enabled) for enabled in active_layers) - 1
        return (layer_number - 1) // int(moe_layer_freq)

    def set_layer_number(self, layer_number: int) -> None:
        self.layer_number = layer_number
        moe_ordinal = self._get_moe_ordinal(layer_number)
        self._is_first_moe_layer = moe_ordinal == 0
        self._is_block_start = (moe_ordinal % self.block_size) == 0
        block_idx = moe_ordinal // self.block_size
        key = (id(self.config), block_idx)
        shared_block = self._shared_path_blocks.get(key)
        if shared_block is None:
            shared_block = _GeoPathBlock(self.config, self.num_cells)
            self._shared_path_blocks[key] = shared_block
        self.path_block = shared_block

        self.state_init.weight.requires_grad_(self._is_first_moe_layer)
        self.state_init.bias.requires_grad_(self._is_first_moe_layer)

    def set_routing_state(self, state: Optional[torch.Tensor]) -> None:
        self._incoming_state = state

    def get_routing_state(self) -> Optional[torch.Tensor]:
        return self._outgoing_state

    def _num_layers_for_logging(self) -> int:
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        return num_layers

    def _save_metric(self, name: str, value: torch.Tensor) -> None:
        if self.layer_number is None:
            return
        save_to_aux_losses_tracker(
            name,
            value,
            self.layer_number,
            self._num_layers_for_logging(),
            reduce_group=self.tp_cp_group,
        )

    def _allreduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            dist.is_available()
            and dist.is_initialized()
            and self.tp_cp_group is not None
            and self.tp_cp_group.size() > 1
        ):
            dist.all_reduce(tensor, group=self.tp_cp_group)
        return tensor

    def _attach_aux_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        valid_token_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if aux_loss_coeff <= 0:
            return activation
        self._save_metric(aux_loss_name, aux_loss / aux_loss_coeff)
        if self.calculate_per_token_loss:
            if valid_token_count is None:
                valid_token_count = torch.tensor(activation.shape[0], device=activation.device)
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * valid_token_count)
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    @staticmethod
    def _wrap_coordinates(coords: torch.Tensor) -> torch.Tensor:
        return torch.remainder(coords, 1.0)

    @staticmethod
    def _pack_state(position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return torch.cat((position, velocity), dim=-1)

    @staticmethod
    def _unpack_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return state[..., :2], state[..., 2:]

    def _build_fourier_features(self, position: torch.Tensor) -> torch.Tensor:
        self._ensure_path_block()
        frequencies = self.path_block.fourier_frequencies.to(
            device=position.device, dtype=position.dtype
        )
        angles = position.unsqueeze(-1) * frequencies.view(1, 1, -1)
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        return torch.cat(
            (
                sin_features[:, 0, :],
                cos_features[:, 0, :],
                sin_features[:, 1, :],
                cos_features[:, 1, :],
            ),
            dim=-1,
        )

    def _compute_content_logits(self, input: torch.Tensor) -> torch.Tensor:
        if self.content_weight.device.type == 'cpu':
            self.content_weight.data = self.content_weight.data.to(device=torch.cuda.current_device())
        if self.content_bias is not None and self.content_bias.device.type == 'cpu':
            self.content_bias.data = self.content_bias.data.to(device=torch.cuda.current_device())

        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        return router_gating_linear(input, self.content_weight, self.content_bias, router_dtype)

    def _compute_cell_prior(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_path_block()
        cell_positions = self.path_block.cell_positions.to(device=position.device, dtype=position.dtype)
        delta = position.unsqueeze(1) - cell_positions.unsqueeze(0)
        delta = delta - torch.round(delta)
        dist_sq = torch.sum(delta * delta, dim=-1)
        cell_logits = -dist_sq / self.cell_temperature
        cell_probs = torch.softmax(cell_logits, dim=-1).to(dtype=position.dtype)
        expert_prior = cell_probs.index_select(1, self.expert_to_cell.to(position.device))
        return cell_probs, expert_prior

    def _topk_routing_from_scores(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = scores.shape[0]
        top_scores, top_indices = torch.topk(scores, k=self.topk, dim=-1)
        top_probs = torch.softmax(top_scores, dim=-1).to(dtype=scores.dtype)

        routing_probs = scores.new_zeros((num_tokens, self.num_experts))
        rows = torch.arange(num_tokens, device=scores.device).unsqueeze(1).expand_as(top_indices)
        routing_probs.index_put_(
            (rows.reshape(-1), top_indices.reshape(-1)),
            top_probs.reshape(-1),
            accumulate=False,
        )
        routing_map = torch.zeros_like(routing_probs, dtype=torch.bool)
        routing_map.index_put_(
            (rows.reshape(-1), top_indices.reshape(-1)),
            torch.ones(top_indices.numel(), device=scores.device, dtype=torch.bool),
            accumulate=False,
        )
        primary_expert = top_indices[:, 0]
        primary_cell = self.expert_to_cell.to(scores.device)[primary_expert]
        return routing_probs, routing_map, primary_expert, primary_cell

    def _apply_padding_mask(
        self,
        routing_probs: torch.Tensor,
        routing_map: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if padding_mask is None:
            return routing_probs, routing_map, None
        valid_mask = ~padding_mask
        routing_probs = routing_probs * valid_mask.unsqueeze(-1).to(dtype=routing_probs.dtype)
        routing_map = routing_map & valid_mask.unsqueeze(-1)
        return routing_probs, routing_map, valid_mask

    def _filter_valid_rows(
        self, values: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_mask is not None:
            valid_token_count = valid_mask.sum()
            if valid_token_count.item() == 0:
                return values[:0], valid_token_count
            return values[valid_mask], valid_token_count
        valid_token_count = torch.tensor(values.shape[0], device=values.device)
        return values, valid_token_count

    def _apply_cell_balance_loss(
        self, routing_probs: torch.Tensor, cell_probs: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.balance_coeff <= 0:
            return routing_probs
        valid_cell_probs, valid_token_count = self._filter_valid_rows(cell_probs, valid_mask)
        if valid_token_count.item() == 0:
            return routing_probs

        soft_mass = self._allreduce_sum(valid_cell_probs.sum(dim=0))
        total_mass = soft_mass.sum().clamp_min(1e-9)
        mass_dist = soft_mass / total_mass
        uniform_log_prob = -math.log(self.num_cells)
        balance_loss = torch.sum(
            mass_dist * (torch.log(mass_dist.clamp_min(1e-9)) - uniform_log_prob)
        ) * self.balance_coeff
        return self._attach_aux_loss(
            routing_probs,
            self.balance_coeff,
            balance_loss,
            'geopath_balance_loss',
            valid_token_count=valid_token_count,
        )

    def _apply_velocity_loss(
        self, routing_probs: torch.Tensor, velocity: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.velocity_coeff <= 0:
            return routing_probs
        valid_velocity, valid_token_count = self._filter_valid_rows(velocity, valid_mask)
        if valid_token_count.item() == 0:
            return routing_probs
        velocity_loss = torch.sum(valid_velocity * valid_velocity, dim=-1).mean() * self.velocity_coeff
        return self._attach_aux_loss(
            routing_probs,
            self.velocity_coeff,
            velocity_loss,
            'geopath_velocity_loss',
            valid_token_count=valid_token_count,
        )

    def _apply_acceleration_loss(
        self,
        routing_probs: torch.Tensor,
        previous_velocity: torch.Tensor,
        current_velocity: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.acceleration_coeff <= 0:
            return routing_probs
        acceleration = current_velocity - previous_velocity
        valid_acceleration, valid_token_count = self._filter_valid_rows(acceleration, valid_mask)
        if valid_token_count.item() == 0:
            return routing_probs
        acceleration_loss = (
            torch.sum(valid_acceleration * valid_acceleration, dim=-1).mean()
            * self.acceleration_coeff
        )
        return self._attach_aux_loss(
            routing_probs,
            self.acceleration_coeff,
            acceleration_loss,
            'geopath_acceleration_loss',
            valid_token_count=valid_token_count,
        )

    def _apply_turn_loss(
        self,
        routing_probs: torch.Tensor,
        previous_velocity: torch.Tensor,
        current_velocity: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.turn_coeff <= 0:
            return routing_probs
        prev_vel, valid_token_count = self._filter_valid_rows(previous_velocity, valid_mask)
        curr_vel, valid_token_count = self._filter_valid_rows(current_velocity, valid_mask)
        if valid_token_count.item() == 0:
            return routing_probs
        cosine = F.cosine_similarity(prev_vel, curr_vel, dim=-1, eps=1e-6)
        turn_loss = F.relu(-cosine).mean() * self.turn_coeff
        return self._attach_aux_loss(
            routing_probs,
            self.turn_coeff,
            turn_loss,
            'geopath_turn_loss',
            valid_token_count=valid_token_count,
        )

    def _log_metrics(
        self,
        cell_probs: torch.Tensor,
        primary_cells: torch.Tensor,
        primary_experts: torch.Tensor,
        previous_velocity: torch.Tensor,
        current_velocity: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> None:
        valid_cell_probs, valid_token_count = self._filter_valid_rows(cell_probs, valid_mask)
        if valid_token_count.item() == 0:
            return
        valid_prev_vel, _ = self._filter_valid_rows(previous_velocity, valid_mask)
        valid_curr_vel, _ = self._filter_valid_rows(current_velocity, valid_mask)
        if valid_mask is not None:
            active_cells = primary_cells[valid_mask]
            active_experts = primary_experts[valid_mask]
        else:
            active_cells = primary_cells
            active_experts = primary_experts

        soft_mass = self._allreduce_sum(valid_cell_probs.sum(dim=0))
        total_mass = soft_mass.sum().clamp_min(1e-9)
        mass_dist = soft_mass / total_mass
        cell_entropy = -(mass_dist * torch.log(mass_dist.clamp_min(1e-9))).sum()

        speed = torch.linalg.vector_norm(valid_curr_vel, dim=-1)
        acceleration = torch.linalg.vector_norm(valid_curr_vel - valid_prev_vel, dim=-1)
        avg_speed = self._allreduce_sum(speed.sum()) / self._allreduce_sum(
            torch.tensor(speed.shape[0], device=speed.device, dtype=torch.float32)
        ).clamp_min(1.0)
        avg_acceleration = self._allreduce_sum(acceleration.sum()) / self._allreduce_sum(
            torch.tensor(acceleration.shape[0], device=acceleration.device, dtype=torch.float32)
        ).clamp_min(1.0)

        cosine = F.cosine_similarity(valid_prev_vel, valid_curr_vel, dim=-1, eps=1e-6)
        u_turn_rate = self._allreduce_sum((cosine < 0).float().sum()) / self._allreduce_sum(
            torch.tensor(cosine.shape[0], device=cosine.device, dtype=torch.float32)
        ).clamp_min(1.0)

        distinct_cells = torch.zeros(self.num_cells, device=active_cells.device, dtype=torch.float32)
        distinct_cells[torch.unique(active_cells)] = 1.0
        distinct_cells = self._allreduce_sum(distinct_cells).sum()

        distinct_experts = torch.zeros(
            self.num_experts, device=active_experts.device, dtype=torch.float32
        )
        distinct_experts[torch.unique(active_experts)] = 1.0
        distinct_experts = self._allreduce_sum(distinct_experts).sum()

        self._save_metric('geopath_cell_entropy', cell_entropy)
        self._save_metric('geopath_avg_speed', avg_speed)
        self._save_metric('geopath_avg_acceleration', avg_acceleration)
        self._save_metric('geopath_u_turn_rate', u_turn_rate)
        self._save_metric('geopath_distinct_cells', distinct_cells)
        self._save_metric('geopath_distinct_experts', distinct_experts)

    def forward(self, input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        self._ensure_path_block()
        seq_length, bsz = input.shape[:2]
        flat_input = input.reshape(-1, input.shape[-1])
        flat_padding_mask = padding_mask.reshape(-1).bool() if padding_mask is not None else None

        previous_state = self._incoming_state
        if (
            previous_state is None
            or previous_state.shape[:2] != input.shape[:2]
            or previous_state.shape[-1] != 4
        ):
            previous_position = torch.sigmoid(self.state_init(flat_input))
            previous_velocity = torch.zeros_like(previous_position)
        else:
            previous_flat = previous_state.reshape(-1, 4).to(device=flat_input.device, dtype=flat_input.dtype)
            previous_position, previous_velocity = self._unpack_state(previous_flat)

        if self._is_block_start and not self._is_first_moe_layer:
            reset_gate = torch.sigmoid(self.path_block.position_reset_gate(flat_input))
            reset_position = torch.sigmoid(self.path_block.position_reset(flat_input))
            previous_position = self._wrap_coordinates(
                (1.0 - reset_gate) * previous_position + reset_gate * reset_position
            )

        path_inputs = torch.cat(
            (flat_input, self._build_fourier_features(previous_position), previous_velocity), dim=-1
        )
        raw_delta = torch.tanh(self.path_block.velocity_update(path_inputs))
        update_gate = torch.sigmoid(self.path_block.velocity_gate(path_inputs))
        current_velocity = previous_velocity + self.eta * update_gate * raw_delta
        current_position = self._wrap_coordinates(previous_position + current_velocity)
        current_flat = self._pack_state(current_position, current_velocity)
        self._outgoing_state = current_flat.view(seq_length, bsz, 4)

        cell_probs, expert_prior = self._compute_cell_prior(current_position)
        content_logits = self._compute_content_logits(flat_input)
        fused_scores = content_logits + self.prior_weight * torch.log(expert_prior.clamp_min(1e-9))
        routing_probs, routing_map, primary_expert, primary_cell = self._topk_routing_from_scores(
            fused_scores
        )
        routing_probs, routing_map, valid_mask = self._apply_padding_mask(
            routing_probs, routing_map, flat_padding_mask
        )

        self._log_metrics(
            cell_probs,
            primary_cell,
            primary_expert,
            previous_velocity,
            current_velocity,
            valid_mask,
        )
        if self.training and torch.is_grad_enabled():
            routing_probs = self._apply_cell_balance_loss(routing_probs, cell_probs, valid_mask)
            routing_probs = self._apply_velocity_loss(routing_probs, current_velocity, valid_mask)
            routing_probs = self._apply_acceleration_loss(
                routing_probs, previous_velocity, current_velocity, valid_mask
            )
            routing_probs = self._apply_turn_loss(
                routing_probs, previous_velocity, current_velocity, valid_mask
            )

        return routing_probs, routing_map


class StatefulPrototypeRouter(MegatronModule):
    """Stateful prototype router with high-dimensional per-token latent routing states."""

    supports_routing_state = True

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.topk = self.config.moe_router_topk
        self.state_dim = int(self.config.stateful_router_dim)
        self.eta = float(self.config.stateful_router_eta)
        self.balance_coeff = float(self.config.stateful_router_balance_coeff)
        self.temporal_coeff = float(self.config.stateful_router_temporal_coeff)
        self.score_temperature = float(self.config.stateful_router_score_temperature)
        self.layer_number = None
        self.tp_cp_group = pg_collection.tp_cp
        self.calculate_per_token_loss = self.config.calculate_per_token_loss

        router_hidden_size = min(max(max(self.config.hidden_size // 2, self.state_dim), 64), 256)
        mlp_input_dim = self.config.hidden_size + self.state_dim

        self.state_init = torch.nn.Linear(self.config.hidden_size, self.state_dim, bias=True)
        self.state_update = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_dim, router_hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(router_hidden_size, self.state_dim),
        )
        self.state_gate = torch.nn.Linear(mlp_input_dim, self.state_dim, bias=True)
        self.expert_prototypes = torch.nn.Parameter(
            torch.empty((self.num_experts, self.state_dim), dtype=torch.float32)
        )

        self._incoming_state: Optional[torch.Tensor] = None
        self._outgoing_state: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.config.perform_initialization:
            self.config.init_method(self.state_init.weight)
            torch.nn.init.zeros_(self.state_init.bias)
            self.config.init_method(self.state_update[0].weight)
            torch.nn.init.zeros_(self.state_update[0].bias)
            self.config.output_layer_init_method(self.state_update[2].weight)
            torch.nn.init.zeros_(self.state_update[2].bias)
            self.config.init_method(self.state_gate.weight)
            torch.nn.init.zeros_(self.state_gate.bias)
            self.config.init_method(self.expert_prototypes)

        for linear in (self.state_init, self.state_update[0], self.state_update[2], self.state_gate):
            linear.weight.data = linear.weight.data.to(dtype=self.config.params_dtype)
            linear.bias.data = linear.bias.data.to(dtype=self.config.params_dtype)
            setattr(linear.weight, 'sequence_parallel', self.config.sequence_parallel)
            setattr(linear.bias, 'sequence_parallel', self.config.sequence_parallel)
        self.expert_prototypes.data = self.expert_prototypes.data.to(dtype=self.config.params_dtype)

    def set_layer_number(self, layer_number: int) -> None:
        self.layer_number = layer_number
        is_first_moe_layer = True
        moe_layer_freq = self.config.moe_layer_freq
        if isinstance(moe_layer_freq, list):
            first_moe_index = next((idx for idx, enabled in enumerate(moe_layer_freq) if enabled), 0)
            is_first_moe_layer = layer_number == (first_moe_index + 1)
        else:
            is_first_moe_layer = layer_number == 1

        # Only the first active MoE layer initializes state directly from hidden states.
        # Later stateful layers receive routing state from the previous MoE layer,
        # so their state_init parameters would otherwise remain permanently unused and break
        # DDP overlap-grad bookkeeping on the first batch.
        self.state_init.weight.requires_grad_(is_first_moe_layer)
        self.state_init.bias.requires_grad_(is_first_moe_layer)

    def set_routing_state(self, state: Optional[torch.Tensor]) -> None:
        self._incoming_state = state

    def get_routing_state(self) -> Optional[torch.Tensor]:
        return self._outgoing_state

    def _num_layers_for_logging(self) -> int:
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        return num_layers

    def _save_metric(self, name: str, value: torch.Tensor) -> None:
        if self.layer_number is None:
            return
        save_to_aux_losses_tracker(
            name,
            value,
            self.layer_number,
            self._num_layers_for_logging(),
            reduce_group=self.tp_cp_group,
        )

    def _allreduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            dist.is_available()
            and dist.is_initialized()
            and self.tp_cp_group is not None
            and self.tp_cp_group.size() > 1
        ):
            dist.all_reduce(tensor, group=self.tp_cp_group)
        return tensor

    def _attach_aux_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        valid_token_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if aux_loss_coeff <= 0:
            return activation
        self._save_metric(aux_loss_name, aux_loss / aux_loss_coeff)
        if self.calculate_per_token_loss:
            if valid_token_count is None:
                valid_token_count = torch.tensor(activation.shape[0], device=activation.device)
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * valid_token_count)
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return F.normalize(state, dim=-1, eps=1e-6)

    def _compute_stateful_routing(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = state.shape[0]
        normalized_state = self._normalize_state(state)
        normalized_prototypes = self._normalize_state(
            self.expert_prototypes.to(device=state.device, dtype=state.dtype)
        )
        routing_scores = torch.matmul(normalized_state, normalized_prototypes.transpose(0, 1))
        routing_scores = routing_scores / self.score_temperature

        top_scores, top_indices = torch.topk(routing_scores, k=self.topk, dim=-1)
        top_probs = torch.softmax(top_scores, dim=-1).to(dtype=state.dtype)

        routing_probs = state.new_zeros((num_tokens, self.num_experts))
        rows = torch.arange(num_tokens, device=state.device).unsqueeze(1).expand_as(top_indices)
        routing_probs.index_put_(
            (rows.reshape(-1), top_indices.reshape(-1)),
            top_probs.reshape(-1),
            accumulate=False,
        )
        routing_map = torch.zeros_like(routing_probs, dtype=torch.bool)
        routing_map.index_put_(
            (rows.reshape(-1), top_indices.reshape(-1)),
            torch.ones(top_indices.numel(), device=state.device, dtype=torch.bool),
            accumulate=False,
        )
        primary_expert = top_indices[:, 0]
        return routing_probs, routing_map, primary_expert

    def _apply_padding_mask(
        self,
        routing_probs: torch.Tensor,
        routing_map: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if padding_mask is None:
            return routing_probs, routing_map, None
        valid_mask = ~padding_mask
        routing_probs = routing_probs * valid_mask.unsqueeze(-1).to(dtype=routing_probs.dtype)
        routing_map = routing_map & valid_mask.unsqueeze(-1)
        return routing_probs, routing_map, valid_mask

    def _apply_balance_loss(
        self, routing_probs: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.balance_coeff <= 0:
            return routing_probs

        masked_probs = routing_probs
        if valid_mask is not None:
            masked_probs = routing_probs * valid_mask.unsqueeze(-1).to(dtype=routing_probs.dtype)
            valid_token_count = valid_mask.sum()
        else:
            valid_token_count = torch.tensor(routing_probs.shape[0], device=routing_probs.device)

        if valid_token_count.item() == 0:
            return routing_probs

        soft_mass = self._allreduce_sum(masked_probs.sum(dim=0))
        total_mass = soft_mass.sum().clamp_min(1e-9)
        mass_dist = soft_mass / total_mass
        uniform_log_prob = -math.log(self.num_experts)
        balance_loss = torch.sum(
            mass_dist * (torch.log(mass_dist.clamp_min(1e-9)) - uniform_log_prob)
        ) * self.balance_coeff
        return self._attach_aux_loss(
            routing_probs,
            self.balance_coeff,
            balance_loss,
            'stateful_balance_loss',
            valid_token_count=valid_token_count,
        )

    def _apply_temporal_loss(
        self,
        routing_probs: torch.Tensor,
        previous_state: torch.Tensor,
        current_state: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.temporal_coeff <= 0:
            return routing_probs

        state_delta = current_state - previous_state
        delta_sq = torch.sum(state_delta * state_delta, dim=-1)
        if valid_mask is not None:
            valid_token_count = valid_mask.sum()
            if valid_token_count.item() == 0:
                return routing_probs
            delta_sq = delta_sq[valid_mask]
        else:
            valid_token_count = torch.tensor(delta_sq.shape[0], device=delta_sq.device)

        temporal_loss = delta_sq.mean() * self.temporal_coeff
        return self._attach_aux_loss(
            routing_probs,
            self.temporal_coeff,
            temporal_loss,
            'stateful_temporal_loss',
            valid_token_count=valid_token_count,
        )

    def _log_metrics(
        self,
        routing_probs: torch.Tensor,
        expert_ids: torch.Tensor,
        previous_state: torch.Tensor,
        current_state: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> None:
        if valid_mask is not None:
            valid_token_count = valid_mask.sum()
            if valid_token_count.item() == 0:
                return
            masked_probs = routing_probs * valid_mask.unsqueeze(-1).to(dtype=routing_probs.dtype)
            active_expert_ids = expert_ids[valid_mask]
            state_delta = current_state[valid_mask] - previous_state[valid_mask]
        else:
            masked_probs = routing_probs
            active_expert_ids = expert_ids
            state_delta = current_state - previous_state

        soft_mass = self._allreduce_sum(masked_probs.sum(dim=0))
        total_mass = soft_mass.sum().clamp_min(1e-9)
        mass_dist = soft_mass / total_mass
        occupancy_entropy = -(mass_dist * torch.log(mass_dist.clamp_min(1e-9))).sum()
        delta_norm_sum = torch.linalg.vector_norm(state_delta, dim=-1).sum()
        delta_count = torch.tensor(
            state_delta.shape[0], device=state_delta.device, dtype=torch.float32
        )
        delta_norm_sum = self._allreduce_sum(delta_norm_sum)
        delta_count = self._allreduce_sum(delta_count)
        avg_state_delta = delta_norm_sum / delta_count.clamp_min(1.0)

        active_experts = torch.zeros(self.num_experts, device=expert_ids.device, dtype=torch.float32)
        active_experts[torch.unique(active_expert_ids)] = 1.0
        active_experts = self._allreduce_sum(active_experts)
        distinct_experts = active_experts.sum()

        self._save_metric('stateful_occupancy_entropy', occupancy_entropy)
        self._save_metric('stateful_avg_state_delta', avg_state_delta)
        self._save_metric('stateful_distinct_experts', distinct_experts)

    def forward(self, input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        seq_length, bsz = input.shape[:2]
        flat_input = input.reshape(-1, input.shape[-1])
        flat_padding_mask = padding_mask.reshape(-1).bool() if padding_mask is not None else None

        previous_state = self._incoming_state
        if previous_state is None or previous_state.shape[:2] != input.shape[:2]:
            previous_state = self._normalize_state(self.state_init(flat_input)).view(
                seq_length, bsz, self.state_dim
            )
        else:
            previous_state = previous_state.to(device=flat_input.device, dtype=flat_input.dtype)

        previous_flat = previous_state.reshape(-1, self.state_dim).to(dtype=flat_input.dtype)
        state_inputs = torch.cat((flat_input, previous_flat), dim=-1)
        raw_delta = torch.tanh(self.state_update(state_inputs))
        update_gate = torch.sigmoid(self.state_gate(state_inputs))
        current_flat = self._normalize_state(previous_flat + self.eta * update_gate * raw_delta)
        self._outgoing_state = current_flat.view(seq_length, bsz, self.state_dim)

        routing_probs, routing_map, expert_ids = self._compute_stateful_routing(current_flat)
        routing_probs, routing_map, valid_mask = self._apply_padding_mask(
            routing_probs, routing_map, flat_padding_mask
        )

        self._log_metrics(routing_probs, expert_ids, previous_flat, current_flat, valid_mask)
        if self.training and torch.is_grad_enabled():
            routing_probs = self._apply_balance_loss(routing_probs, valid_mask)
            routing_probs = self._apply_temporal_loss(
                routing_probs, previous_flat, current_flat, valid_mask
            )

        return routing_probs, routing_map


class TopKRouter(Router):
    """Route each token to the top-k experts.

    The workflow of TopKRouter is as follows:
    (1) Calculate the logits by the router gating network.
    (2) Calculate the routing probabilities and map for top-k selection with score function.
    (3) [Optional] Apply token dropping to top-k expert selection.
    (4) [Optional] Apply the auxiliary load balancing loss for the given scores and routing map.

    Naming convention:
        logits: The output logits by the router gating network.
        scores: The scores after score function used to select the experts and calculate aux loss.
        probs: The topk weights used to combined the experts' outputs.
        routing_map: The masked routing map between tokens and experts.
    """

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

        # Initialize global tokens per expert for global aux loss
        if self.get_aux_loss_coeff("global_aux_loss") > 0:
            self.register_buffer(
                'global_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'ga_steps',
                torch.tensor(0, dtype=torch.float32, device=torch.cuda.current_device()),
                persistent=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

        self.router_replay = None
        if self.config.moe_enable_routing_replay:
            self.router_replay = RouterReplay()

        self.matryoshka_moe = bool(getattr(self.config, "matryoshka_moe", False))
        self.matryoshka_tau = float(getattr(self.config, "matryoshka_tau", 1.0))
        self.matryoshka_k = None
        self._matryoshka_k_values = None
        self._matryoshka_sampling_probs = None
        self.dynamic_topk = bool(getattr(self.config, "moe_dynamic_topk", False))
        self.dynamic_topk_mode = str(getattr(self.config, "moe_dynamic_topk_mode", "random_gauss"))
        dynamic_topk_min = getattr(self.config, "moe_dynamic_topk_min", None)
        dynamic_topk_max = getattr(self.config, "moe_dynamic_topk_max", None)
        self.dynamic_topk_min = (
            int(dynamic_topk_min) if dynamic_topk_min is not None else max(1, self.topk - 2)
        )
        self.dynamic_topk_max = (
            int(dynamic_topk_max)
            if dynamic_topk_max is not None
            else self.config.num_moe_experts
            if self.topk + 2 > self.config.num_moe_experts
            else self.topk + 2
        )
        self.dynamic_topk_sigma = float(getattr(self.config, "moe_dynamic_topk_sigma", 1.5))
        self.router_logit_noise = bool(getattr(self.config, "moe_router_logit_noise", False))
        self.router_logit_noise_std = float(
            getattr(self.config, "moe_router_logit_noise_std", 0.0)
        )
        self.default_moe_dense_probs = None
        if self.dynamic_topk:
            candidates = torch.arange(
                self.dynamic_topk_min,
                self.dynamic_topk_max + 1,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            weights = torch.exp(
                -torch.pow(candidates - float(self.topk), 2)
                / (2.0 * (self.dynamic_topk_sigma**2))
            )
            probs = weights / weights.sum()
            self.register_buffer(
                "_dynamic_topk_candidates", candidates.to(dtype=torch.int64), persistent=False
            )
            self.register_buffer("_dynamic_topk_sampling_probs", probs, persistent=False)
        if self.matryoshka_moe:
            requested_k = getattr(self.config, "matryoshka_k", None) or [self.config.num_moe_experts]
            self.matryoshka_k = sorted(
                {
                    int(k)
                    for k in requested_k
                    if self.topk <= int(k) <= self.config.num_moe_experts
                }
            )
            if not self.matryoshka_k:
                raise ValueError(
                    "Matryoshka MoE requires at least one valid k satisfying "
                    "moe_router_topk <= k <= num_moe_experts."
                )
            if self.matryoshka_tau <= 0:
                raise ValueError("matryoshka_tau must be > 0 when matryoshka_moe is enabled.")

            k_tensor = torch.tensor(
                self.matryoshka_k,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            scores = torch.log(k_tensor)
            self.register_buffer(
                "_matryoshka_k_values",
                k_tensor.to(dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "_matryoshka_sampling_probs",
                torch.softmax(scores / self.matryoshka_tau, dim=0),
                persistent=False,
            )

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def _sample_dynamic_topk(self, num_tokens: int, device: torch.device) -> Tuple[torch.Tensor, int]:
        """Sample per-token k for RandomGauss dynamic top-k."""
        if not self.dynamic_topk:
            k_per_token = torch.full((num_tokens,), self.topk, device=device, dtype=torch.long)
            return k_per_token, self.topk

        if self.training:
            sampled_indices = torch.multinomial(
                self._dynamic_topk_sampling_probs.expand(num_tokens, -1),
                num_samples=1,
                replacement=True,
            ).squeeze(-1)
            k_per_token = self._dynamic_topk_candidates[sampled_indices]
            current_topk = self.dynamic_topk_max
        else:
            k_per_token = torch.full((num_tokens,), self.topk, device=device, dtype=torch.long)
            current_topk = self.topk
        return k_per_token, current_topk

    def _router_logit_noise_enabled(self) -> bool:
        """Whether to add training-time noise before fixed top-k expert selection."""
        return self.training and self.router_logit_noise and self.router_logit_noise_std > 0.0

    def _expert_dropout_p(self) -> float:
        return float(os.environ.get("QWEN3_EXPERT_DROPOUT_P", "0.0") or 0.0)

    def _expert_dropout_enabled(self) -> bool:
        """Whether to mask random experts before fixed top-k expert selection."""
        return self.training and not self.dynamic_topk and self._expert_dropout_p() > 0.0

    def _default_moe_enabled(self) -> bool:
        return os.environ.get("QWEN3_DEFAULT_MOE", "0").lower() in ("1", "true", "yes", "on")

    def _prepare_default_moe_dense_probs(
        self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> None:
        """Expose dense router probabilities for Default MoE's non-selected expert path."""
        self.default_moe_dense_probs = None
        if not self._default_moe_enabled():
            return
        if self.routing_type == "sinkhorn":
            raise ValueError("Default MoE only supports ordinary fixed TopK routing, not sinkhorn.")
        if self.dynamic_topk or self._router_logit_noise_enabled() or self._expert_dropout_enabled():
            raise ValueError(
                "Default MoE is an independent baseline; do not combine it with dynamic top-k, "
                "router logit noise, or expert dropout."
            )
        if self.score_function != "softmax" or not self.config.moe_router_pre_softmax:
            raise ValueError("Default MoE requires softmax routing with --moe-router-pre-softmax.")

        dense_probs = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        if self.config.moe_router_topk_scaling_factor:
            dense_probs = dense_probs * self.config.moe_router_topk_scaling_factor
        if padding_mask is not None:
            dense_probs = dense_probs * (~padding_mask).unsqueeze(-1).to(dtype=dense_probs.dtype)
        self.default_moe_dense_probs = dense_probs

    def _dynamic_random_gauss_routing(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute token-level dynamic top-k routing for the ordinary TopKRouter."""
        num_tokens = logits.shape[0]
        device = logits.device
        k_per_token, current_topk = self._sample_dynamic_topk(num_tokens, device)

        if self.score_function == "softmax":
            if self.config.moe_router_pre_softmax:
                scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
                routing_weights, selected_experts = torch.topk(scores, current_topk, dim=-1)
            else:
                topk_logits, selected_experts = torch.topk(logits, current_topk, dim=-1)
                routing_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32).type_as(
                    logits
                )
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
            routing_weights, selected_experts = torch.topk(scores, current_topk, dim=-1)
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")

        valid_mask = (
            torch.arange(current_topk, device=device).unsqueeze(0) < k_per_token.unsqueeze(1)
        )
        routing_weights = routing_weights * valid_mask.to(dtype=routing_weights.dtype)
        if not (self.score_function == "softmax" and self.config.moe_router_pre_softmax):
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
        if self.config.moe_router_topk_scaling_factor:
            routing_weights = routing_weights * self.config.moe_router_topk_scaling_factor

        routing_probs = torch.zeros_like(logits).scatter(1, selected_experts, routing_weights)
        routing_map = (
            torch.zeros_like(logits)
            .int()
            .scatter(1, selected_experts, valid_mask.to(dtype=torch.int32))
            .bool()
        )
        return routing_probs, routing_map, selected_experts, valid_mask

    def _topk_routing_with_logit_noise(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed top-k routing where Gaussian noise affects selection only.

        The selected experts are taken from clean logits plus zero-mean Gaussian noise, while
        dispatch probabilities are still computed from the clean logits. This keeps eval and
        inference identical to the normal TopKRouter path because this helper is training-only.
        """
        selection_logits = logits + torch.randn_like(logits) * self.router_logit_noise_std
        _, selected_experts = torch.topk(selection_logits, self.topk, dim=-1)

        if self.score_function == "softmax":
            if self.config.moe_router_pre_softmax:
                scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
                routing_weights = torch.gather(scores, dim=1, index=selected_experts)
            else:
                selected_logits = torch.gather(logits, dim=1, index=selected_experts)
                routing_weights = torch.softmax(
                    selected_logits, dim=-1, dtype=torch.float32
                ).type_as(logits)
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
            routing_weights = torch.gather(scores, dim=1, index=selected_experts)
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")

        if self.config.moe_router_topk_scaling_factor:
            routing_weights = routing_weights * self.config.moe_router_topk_scaling_factor

        routing_probs = torch.zeros_like(logits).scatter(1, selected_experts, routing_weights)
        routing_map = torch.zeros_like(logits).int().scatter(1, selected_experts, 1).bool()
        return routing_probs, routing_map, selected_experts

    def _topk_routing_with_expert_dropout(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed top-k routing where random experts are unavailable for selection only."""
        expert_dropout_p = self._expert_dropout_p()
        if not 0.0 <= expert_dropout_p < 1.0:
            raise ValueError("QWEN3_EXPERT_DROPOUT_P must be in [0, 1).")

        selection_logits = logits.float().clone()
        num_experts = selection_logits.shape[-1]
        drop_mask = torch.rand((num_experts,), device=selection_logits.device) < expert_dropout_p
        if num_experts - int(drop_mask.sum().item()) < self.topk:
            drop_mask.fill_(True)
            keep_indices = torch.randperm(num_experts, device=selection_logits.device)[: self.topk]
            drop_mask[keep_indices] = False
        if (
            not getattr(self, "_expert_dropout_runtime_logged", False)
            and self.layer_number in (0, 1)
            and (
                not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            )
        ):
            print(
                "[EXPERT_DROPOUT_ACTIVE] "
                f"layer={self.layer_number} p={expert_dropout_p} "
                f"dropped_experts={int(drop_mask.sum().item())}/{num_experts} "
                f"topk={self.topk}",
                flush=True,
            )
            self._expert_dropout_runtime_logged = True
        selection_logits[:, drop_mask] = torch.finfo(selection_logits.dtype).min
        _, selected_experts = torch.topk(selection_logits, self.topk, dim=-1)

        if self.score_function == "softmax":
            if self.config.moe_router_pre_softmax:
                scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
                routing_weights = torch.gather(scores, dim=1, index=selected_experts)
            else:
                selected_logits = torch.gather(logits, dim=1, index=selected_experts)
                routing_weights = torch.softmax(
                    selected_logits, dim=-1, dtype=torch.float32
                ).type_as(logits)
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
            routing_weights = torch.gather(scores, dim=1, index=selected_experts)
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")

        if self.config.moe_router_topk_scaling_factor:
            routing_weights = routing_weights * self.config.moe_router_topk_scaling_factor

        routing_probs = torch.zeros_like(logits).scatter(1, selected_experts, routing_weights)
        routing_map = torch.zeros_like(logits).int().scatter(1, selected_experts, 1).bool()
        return routing_probs, routing_map, selected_experts

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def get_aux_loss_coeff(self, aux_loss_type: str) -> float:
        """Return the aux loss coeff for the given auxiliary loss type.
        If the auxiliary loss type is not found, return 0.0.
        """
        if isinstance(self.routing_type, str):
            if self.routing_type == aux_loss_type:
                return self.config.moe_aux_loss_coeff
        if isinstance(self.routing_type, list):
            try:
                idx = self.routing_type.index(aux_loss_type)
                return self.config.moe_aux_loss_coeff[idx]
            except ValueError:
                return 0.0
        return 0.0

    def is_aux_loss_enabled(self) -> bool:
        """Check if the auxiliary loss is enabled."""
        for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
            if self.get_aux_loss_coeff(aux_loss_type) > 0:
                return True
        return False

    def _apply_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        with_padding_mask: bool = False,
        total_assignments: Optional[torch.Tensor] = None,
        local_num_tokens: Optional[torch.Tensor] = None,
        total_num_tokens: Optional[torch.Tensor] = None,
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs

        if total_assignments is None or local_num_tokens is None or total_num_tokens is None:
            global_tokens_per_expert, local_num_tokens, total_num_tokens = (
                get_tokens_per_expert_and_token_count(
                    routing_map=routing_map,
                    reduce_group=self.tp_cp_group,
                    topk=self.topk,
                    with_padding_mask=with_padding_mask,
                )
            )
        else:
            local_tokens_per_expert = routing_map.sum(dim=0).to(dtype=torch.float32)
            global_tokens_per_expert = reduce_from_tensor_model_parallel_region(
                local_tokens_per_expert, self.tp_cp_group
            )

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=aux_loss_coeff,
            fused=self.config.moe_router_fusion,
            total_assignments=total_assignments,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            aux_loss_coeff,
            aux_loss,
            "load_balancing_loss",
            self.tp_cp_group,
            valid_token_count=local_num_tokens,
        )
        return probs

    def _apply_seq_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
        with_padding_mask: bool = False,
    ):
        """Apply the sequence-level auxiliary loss for the given scores and routing map.

        To calculate the sequence-level aux loss, we reshape the batch_size dimension to
        experts dimension. The resulted loss by switch_load_balancing_loss_func is equal
        to the sum of aux loss for each sequence in the batch. And then we divide the aux
        loss by the batch size to get averaged aux loss.
        """
        seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
        if seq_aux_loss_coeff == 0:
            return probs

        scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
        routing_map = routing_map.reshape(seq_length, -1)

        global_tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=self.tp_cp_group,
                with_padding_mask=with_padding_mask,
                topk=self.topk * bsz,
            )
        )

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_for_aux_loss,
                tokens_per_expert=global_tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
                fused=self.config.moe_router_fusion,
            )
            / bsz
        )

        probs = self.attach_and_log_load_balancing_loss(
            probs,
            seq_aux_loss_coeff,
            aux_loss,
            "seq_load_balancing_loss",
            self.tp_cp_group,
            valid_token_count=local_num_tokens,
        )
        return probs

    def _apply_global_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        with_padding_mask: bool = False,
    ):
        """Apply the global auxiliary loss for the given scores and routing map."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        # Use unified function to compute tokens_per_expert and num_tokens
        global_tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=self.tp_dp_cp_group,
                with_padding_mask=with_padding_mask,
                topk=self.topk,
            )
        )

        self.global_tokens_per_expert += global_tokens_per_expert
        self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=averated_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=global_aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            global_aux_loss_coeff,
            global_aux_loss,
            "global_load_balancing_loss",
            self.tp_dp_cp_group,
            reduce_group_has_dp=True,
            valid_token_count=local_num_tokens,
        )
        return probs

    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        reduce_group: torch.distributed.ProcessGroup,
        reduce_group_has_dp: bool = False,
        valid_token_count: Optional[Union[int, torch.Tensor]] = None,
    ):
        """Attach aux loss function to activation and add to logging.

        Args:
            activation (torch.Tensor): Activation tensor to attach the aux loss to.
            aux_loss_coeff (float): Coefficient for the aux loss.
            aux_loss (torch.Tensor): Computed aux loss.
            aux_loss_name (str): Name of the aux loss for logging.
            reduce_group (torch.distributed.ProcessGroup): Process group for reduction.
            reduce_group_has_dp (bool): Whether the reduce group has data parallel ranks.
                Set this to True if the reduce group has data parallel ranks. This flag is used to
                ensure the correct reduction in aux loss tracking.
            valid_token_count (int or torch.Tensor, optional): Number of valid tokens excluding
                padding tokens. Can be a Python int or a torch.Tensor (typically 0-d tensor).
                If None, uses activation.shape[0]. Defaults to None.
        """
        # TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly
        # add the aux loss logging value to other layer's since it is difficult to get the
        # correct layer_number for MTP. It does not affect the correctness of the calculation
        # results and the reduced load_balancing_loss logging value.
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        save_to_aux_losses_tracker(
            aux_loss_name,
            aux_loss / aux_loss_coeff,
            self.layer_number,
            num_layers,
            reduce_group=reduce_group,
            reduce_group_has_dp=reduce_group_has_dp,
        )
        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            # Use valid_token_count (excluding padding) if provided, otherwise use total tokens.
            num_tokens = valid_token_count if valid_token_count is not None else activation.shape[0]
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * num_tokens)
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits, padding_mask: Optional[torch.Tensor] = None):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape in [num_tokens]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = self.config.moe_z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(logits, moe_z_loss_coeff, padding_mask=padding_mask)
            if self.calculate_per_token_loss:
                # The expected final scaling for z_loss gradients is
                # 1/(num_micro_batches * dp_size).
                # After commit 02648000, Megatron started using the number of total tokens
                # to scale gradients under the argument of calculate_per_token_loss,
                # which scales both the main_loss gradient and z_loss gradient by
                # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads().
                # To correct this scaling, we need to scale the z_loss by num_local_tokens here.
                # Count valid tokens: sum of inverted mask (False -> True = valid)
                num_tokens = (~padding_mask).sum() if padding_mask is not None else logits.shape[0]
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss * num_tokens)
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)

            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, dtype=input.dtype, device=input.device),
                    torch.tensor(1.0 + eps, dtype=input.dtype, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    @jit_fuser
    def _apply_expert_bias(
        self, routing_map: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Update expert bias and tokens_per_expert
        Prevent extra local tokens accumulation on evaluation or activation recomputation
        """
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                if padding_mask is not None:
                    routing_map = routing_map & (~padding_mask)
                self.local_tokens_per_expert += routing_map.sum(dim=0)

    def routing(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape [seq_length, bsz]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        if self.matryoshka_moe:
            if self.training:
                sampled_idx = torch.multinomial(self._matryoshka_sampling_probs, 1).item()
                active_expert_count = int(self._matryoshka_k_values[sampled_idx].item())
            else:
                # In eval mode, default to the full model.
                active_expert_count = self.config.num_moe_experts
            if active_expert_count < self.config.num_moe_experts:
                expert_mask = (
                    torch.arange(self.config.num_moe_experts, device=logits.device)
                    >= active_expert_count
                )
                logits = logits.masked_fill(expert_mask, float("-inf"))

        # Flatten padding_mask to [num_tokens] if provided
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits, padding_mask=padding_mask)
        self._prepare_default_moe_dense_probs(logits, padding_mask)

        # Calculate probs and routing_map for token dispatching
        selected_experts = None
        valid_mask = None
        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.dynamic_topk:
            probs, routing_map, selected_experts, valid_mask = self._dynamic_random_gauss_routing(
                logits
            )
        elif self._router_logit_noise_enabled():
            if self._expert_dropout_enabled():
                raise ValueError(
                    "QWEN3_ROUTER_NOISE_STD and QWEN3_EXPERT_DROPOUT_P are independent "
                    "baselines; do not enable both in one run."
                )
            probs, routing_map, selected_experts = self._topk_routing_with_logit_noise(logits)
        elif self._expert_dropout_enabled():
            probs, routing_map, selected_experts = self._topk_routing_with_expert_dropout(logits)
        else:
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=self.config.moe_router_fusion,
                router_replay=self.router_replay,
            )

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Apply each aux loss type and attach aux loss autograd function to probs
        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            # Calculate scores and routing_map for aux loss
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                self.score_function,
                fused=self.config.moe_router_fusion and selected_experts is None,
                padding_mask=padding_mask,
                selected_experts=selected_experts,
                valid_mask=valid_mask,
            )
            total_assignments = None
            local_num_tokens = None
            total_num_tokens = None
            if self.dynamic_topk:
                local_num_tokens = (
                    (~padding_mask).sum()
                    if padding_mask is not None
                    else torch.tensor(logits.shape[0], device=logits.device, dtype=torch.float32)
                ).to(dtype=torch.float32)
                total_num_tokens = reduce_from_tensor_model_parallel_region(
                    local_num_tokens, self.tp_cp_group
                )
                local_total_assignments = routing_map_for_aux_loss.sum().to(dtype=torch.float32)
                total_assignments = reduce_from_tensor_model_parallel_region(
                    local_total_assignments, self.tp_cp_group
                )
            probs = self._apply_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                with_padding_mask=padding_mask is not None,
                total_assignments=total_assignments,
                local_num_tokens=local_num_tokens,
                total_num_tokens=total_num_tokens,
            )
            if not self.dynamic_topk:
                probs = self._apply_seq_aux_loss(
                    probs,
                    scores_for_aux_loss,
                    routing_map_for_aux_loss,
                    seq_length,
                    bsz,
                    with_padding_mask=padding_mask is not None,
                )
                probs = self._apply_global_aux_loss(
                    probs,
                    scores_for_aux_loss,
                    routing_map_for_aux_loss,
                    with_padding_mask=padding_mask is not None,
                )

        # Optionally apply expert bias
        self._apply_expert_bias(routing_map, padding_mask=padding_mask)

        return probs, routing_map

    def reset_global_aux_loss_tracker(self):
        """Reset the global aux loss tracker."""
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def forward(self, input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape [seq_length, bsz]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            # Apply force load balancing with random logits for benchmark
            logits = apply_random_logits(logits)

        probs, routing_map = self.routing(logits, padding_mask=padding_mask)

        return probs, routing_map

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)
