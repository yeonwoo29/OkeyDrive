"""Visibility-modulated Mamba-1 selective SSM candidate refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import rasterized_oriented_iou


def cuda_selective_scan_available() -> bool:
    """Return whether the Mamba selective-scan CUDA extension can be imported."""

    if not torch.cuda.is_available():
        return False
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def selective_scan_reference(
    inputs: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Apply the Mamba-1 recurrence in batch-first logical layout.

    Shapes are inputs/delta ``[B,Z,D_inner]``, A ``[D_inner,D_state]``,
    B/C ``[B,Z,D_state]``, and D ``[D_inner]``. Delta must already be
    positive; no bias or softplus is applied in this function.
    """

    if inputs.shape != delta.shape:
        raise ValueError("inputs and delta must have the same [B,Z,D_inner] shape")
    input_dtype = inputs.dtype
    inputs = inputs.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    D = D.float()
    batch, length, inner = inputs.shape
    state_size = A.shape[-1]
    state = inputs.new_zeros((batch, inner, state_size))
    outputs = []
    for index in range(length):
        step_delta = delta[:, index]
        transition = torch.exp(step_delta[..., None] * A[None])
        input_update = step_delta[..., None] * B[:, index, None, :] * inputs[:, index, :, None]
        state = transition * state + input_update
        output = (state * C[:, index, None, :]).sum(dim=-1) + D * inputs[:, index]
        outputs.append(output)
    return torch.stack(outputs, dim=1).to(input_dtype)


def selective_scan_cuda(
    inputs: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Run the CUDA extension with already-positive, already-modulated delta."""

    if not cuda_selective_scan_available():
        raise RuntimeError("mamba_ssm selective-scan CUDA extension is unavailable")
    if not inputs.is_cuda:
        raise RuntimeError("the CUDA selective scan requires CUDA tensors")
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    output = selective_scan_fn(
        inputs.transpose(1, 2).contiguous(),
        delta.transpose(1, 2).contiguous(),
        A,
        B.transpose(1, 2).contiguous(),
        C.transpose(1, 2).contiguous(),
        D,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    )
    return output.transpose(1, 2)


@dataclass
class MambaDiagnostics:
    delta: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    delta_modulation: torch.Tensor
    B_modulation: torch.Tensor
    C_modulation: torch.Tensor


class VisibilityConditionedMambaBlock(nn.Module):
    """Mamba-1 block whose positive Delta and input-dependent B/C are modulated."""

    def __init__(
        self,
        model_dim: int = 256,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dt_rank: Optional[int] = None,
        visibility_dim: int = 10,
        scan_backend: str = "reference",
        use_modulation: bool = True,
    ):
        super().__init__()
        if scan_backend not in {"reference", "cuda"}:
            raise ValueError("scan_backend must be 'reference' or 'cuda'")
        self.model_dim = model_dim
        self.inner_dim = model_dim * expand
        self.state_dim = state_dim
        self.dt_rank = dt_rank or math.ceil(model_dim / 16)
        self.scan_backend = scan_backend
        self.use_modulation = use_modulation
        self.in_proj = nn.Linear(model_dim, self.inner_dim * 2, bias=False)
        self.conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel,
            groups=self.inner_dim,
            padding=conv_kernel - 1,
        )
        self.ssm_proj = nn.Linear(self.inner_dim, self.dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))[None].repeat(self.inner_dim, 1)
        )
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.context_projection = nn.Linear(model_dim, model_dim)
        self.visibility_conditioner = nn.Sequential(
            nn.Linear(visibility_dim + model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        modulation_input = model_dim * 2
        self.delta_modulator = nn.Linear(modulation_input, self.inner_dim)
        self.B_modulator = nn.Linear(modulation_input, state_dim)
        self.C_modulator = nn.Linear(modulation_input, state_dim)
        self.out_proj = nn.Linear(self.inner_dim, model_dim, bias=False)
        self.norm = nn.LayerNorm(model_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dt_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_std, dt_std)
        dt = torch.exp(
            torch.rand(self.inner_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp_min(1e-4)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def forward(
        self,
        motion_tokens: torch.Tensor,
        visibility: torch.Tensor,
        evidence_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, MambaDiagnostics]:
        if motion_tokens.ndim != 3:
            raise ValueError("motion_tokens must be [B,Z,D]")
        projected, gate = self.in_proj(self.norm(motion_tokens)).chunk(2, dim=-1)
        convolved = self.conv(projected.transpose(1, 2))[..., : motion_tokens.shape[1]]
        inputs = F.silu(convolved.transpose(1, 2))
        parameters = self.ssm_proj(inputs)
        dt_low_rank, B, C = torch.split(
            parameters, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        # Softplus is applied exactly once, before visibility modulation.
        delta = F.softplus(F.linear(dt_low_rank, self.dt_proj.weight, self.dt_proj.bias))
        query = self.context_projection(motion_tokens)
        visible_context = self.visibility_conditioner(torch.cat([visibility, motion_tokens], dim=-1))
        modulation_input = torch.cat([query, visible_context], dim=-1)
        if self.use_modulation:
            evidence = evidence_mask.to(motion_tokens.dtype)
            if evidence.ndim == 2:
                evidence = evidence.unsqueeze(-1)
            s_delta = 0.5 * torch.tanh(self.delta_modulator(modulation_input)) * evidence
            s_B = 0.5 * torch.tanh(self.B_modulator(modulation_input)) * evidence
            s_C = 0.5 * torch.tanh(self.C_modulator(modulation_input)) * evidence
        else:
            s_delta = torch.zeros_like(delta)
            s_B = torch.zeros_like(B)
            s_C = torch.zeros_like(C)
        delta_hat = delta * (1.0 + s_delta)
        B_hat = B * (1.0 + s_B)
        C_hat = C * (1.0 + s_C)
        A = -torch.exp(self.A_log.float())
        skip = self.D.float()
        if self.scan_backend == "cuda":
            scanned = selective_scan_cuda(inputs, delta_hat, A, B_hat, C_hat, skip)
        else:
            scanned = selective_scan_reference(inputs, delta_hat, A, B_hat, C_hat, skip)
        output = motion_tokens + self.out_proj(scanned * F.silu(gate))
        if not return_diagnostics:
            return output
        return output, MambaDiagnostics(delta_hat, B_hat, C_hat, s_delta, s_B, s_C)


def canonical_candidate_order(
    trajectories: torch.Tensor,
    stable_anchor_ids: torch.Tensor,
    left_unit: Tuple[float, float] = (-1.0, 0.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort candidates left-to-right and return permutation plus its inverse.

    DiffusionDrive represents the forward direction as +y in its planning
    decoder, so the ego-left unit vector is (-1, 0). Positive-left lateral
    coordinates are sorted descending. Stable anchor IDs break exact ties.
    """

    if trajectories.ndim != 4:
        raise ValueError("trajectories must be [B,Z,T,2]")
    batch, candidates = trajectories.shape[:2]
    if stable_anchor_ids.ndim == 1:
        stable_anchor_ids = stable_anchor_ids[None].expand(batch, -1)
    if stable_anchor_ids.shape != (batch, candidates):
        raise ValueError("stable anchor IDs must match [B,Z]")
    left = trajectories.new_tensor(left_unit)
    lateral = torch.einsum("bzd,d->bz", trajectories[..., -1, :], left)
    id_order = torch.argsort(stable_anchor_ids, dim=-1, stable=True)
    lateral_by_id = torch.gather(lateral, 1, id_order)
    lateral_order = torch.argsort(lateral_by_id, dim=-1, descending=True, stable=True)
    permutation = torch.gather(id_order, 1, lateral_order)
    inverse = torch.empty_like(permutation)
    positions = torch.arange(candidates, device=trajectories.device)[None].expand(batch, -1)
    inverse.scatter_(1, permutation, positions)
    return permutation, inverse


def apply_candidate_permutation(value: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Apply one ``[B,Z]`` candidate permutation to any ``[B,Z,...]`` tensor."""

    index = permutation
    for _ in range(value.ndim - 2):
        index = index.unsqueeze(-1)
    return torch.gather(value, 1, index.expand_as(value))


class CandidateVisibilityAggregator(nn.Module):
    """Pool class-separated predicted visibility with rasterized BEV IoU weights."""

    def __init__(self, x_bound=(-15.0, 15.0, 0.5), y_bound=(-10.0, 50.0, 0.5)):
        super().__init__()
        self.x_bound = x_bound
        self.y_bound = y_bound

    def forward(
        self,
        trajectories: torch.Tensor,
        object_boxes: torch.Tensor,
        object_visibility: torch.Tensor,
        object_class: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if object_boxes.shape[1] == 0:
            shape = trajectories.shape[:2]
            return (
                trajectories.new_zeros((*shape, 8)),
                torch.zeros((*shape, 2), device=trajectories.device, dtype=torch.bool),
                trajectories.new_zeros((*shape, trajectories.shape[2], 0)),
            )
        iou = rasterized_oriented_iou(
            trajectories,
            object_boxes,
            object_valid,
            x_bound=self.x_bound,
            y_bound=self.y_bound,
        )
        # Later waypoints receive larger weights while retaining all six steps.
        time_weight = torch.linspace(1.0, 2.0, trajectories.shape[2], device=trajectories.device)
        weights = iou * time_weight[None, None, :, None]
        pooled_parts = []
        evidence = []
        for class_index in range(2):
            class_mask = (object_class == class_index) & object_valid
            class_weights = weights * class_mask[:, None, None, :].to(weights.dtype)
            denominator = class_weights.sum(dim=(-1, -2))
            numerator = torch.einsum("bzto,bop->bzp", class_weights, object_visibility)
            pooled_parts.append(numerator / denominator.clamp_min(1e-6).unsqueeze(-1))
            evidence.append(denominator > 0)
        visibility = torch.cat(pooled_parts, dim=-1)
        evidence_mask = torch.stack(evidence, dim=-1)
        return visibility, evidence_mask, iou


class ContextMotionMambaRefiner(nn.Module):
    """Expand command anchors to ten candidates and replace diffusion refinement."""

    def __init__(
        self,
        model_dim: int = 256,
        future_steps: int = 6,
        base_modes: int = 6,
        num_candidates: int = 10,
        state_dim: int = 16,
        scan_backend: str = "reference",
        use_modulation: bool = True,
        order_mode: str = "canonical",
        x_bound=(-15.0, 15.0, 0.5),
        y_bound=(-10.0, 50.0, 0.5),
    ):
        super().__init__()
        if order_mode not in {"canonical", "reversed"}:
            raise ValueError("order_mode must be canonical or reversed")
        self.model_dim = model_dim
        self.future_steps = future_steps
        self.base_modes = base_modes
        self.num_candidates = num_candidates
        self.order_mode = order_mode
        centers = torch.linspace(0, base_modes - 1, num_candidates)
        source = torch.arange(base_modes, dtype=torch.float32)
        initial_mix = -((centers[:, None] - source[None]) ** 2) / 0.5
        self.anchor_mix_logits = nn.Parameter(initial_mix)
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(future_steps * 2, model_dim), nn.GELU(), nn.LayerNorm(model_dim)
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.visibility_aggregator = CandidateVisibilityAggregator(x_bound=x_bound, y_bound=y_bound)
        self.mamba = VisibilityConditionedMambaBlock(
            model_dim=model_dim,
            state_dim=state_dim,
            visibility_dim=10,
            scan_backend=scan_backend,
            use_modulation=use_modulation,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, future_steps * 2)
        )
        self.score_head = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.x_bound = x_bound
        self.y_bound = y_bound

    def _sample_bev(self, bev: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        x = 2.0 * (trajectories[..., 0] - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0]) - 1.0
        y = 2.0 * (trajectories[..., 1] - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0]) - 1.0
        grid = torch.stack([x, y], dim=-1)
        sampled = F.grid_sample(bev, grid, align_corners=True, padding_mode="zeros")
        return sampled.mean(dim=-1).transpose(1, 2)

    def forward(
        self,
        base_trajectories: torch.Tensor,
        navigation_features: torch.Tensor,
        sparse_context: torch.Tensor,
        bev: Optional[torch.Tensor],
        object_boxes: torch.Tensor,
        object_visibility: torch.Tensor,
        object_class: torch.Tensor,
        object_valid: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """Refine candidates for all three route commands without GT inputs.

        Base trajectories are cumulative ``[B,3,6,6,2]`` anchors. Outputs are
        incremental trajectories ``[B,3,10,6,2]`` and scores ``[B,3,10]``.
        """

        batch, commands, modes, steps = base_trajectories.shape[:4]
        if (modes, steps) != (self.base_modes, self.future_steps):
            raise ValueError("base anchor shape does not match configured modes and steps")
        mix = self.anchor_mix_logits.softmax(dim=-1)
        expanded = torch.einsum("zm,bcmtd->bcztd", mix, base_trajectories)
        expanded_nav = torch.einsum("zm,bcmd->bczd", mix, navigation_features)
        flat_trajectories = expanded.reshape(batch * commands, self.num_candidates, steps, 2)
        stable_ids = torch.arange(self.num_candidates, device=expanded.device)
        permutation, inverse = canonical_candidate_order(flat_trajectories, stable_ids)
        if self.order_mode == "reversed":
            permutation = permutation.flip(dims=(1,))
            inverse = torch.empty_like(permutation)
            positions = torch.arange(
                self.num_candidates, device=permutation.device
            )[None].expand_as(permutation)
            inverse.scatter_(1, permutation, positions)
        sorted_trajectories = apply_candidate_permutation(flat_trajectories, permutation)
        flat_nav = apply_candidate_permutation(
            expanded_nav.reshape(batch * commands, self.num_candidates, self.model_dim), permutation
        )
        if bev is None:
            bev_context = torch.zeros_like(flat_nav)
        else:
            repeated_bev = bev[:, None].expand(-1, commands, -1, -1, -1).reshape(
                batch * commands, bev.shape[1], bev.shape[2], bev.shape[3]
            )
            bev_context = self._sample_bev(repeated_bev, sorted_trajectories)
        repeated_sparse = sparse_context[:, None, None].expand(
            -1, commands, self.num_candidates, -1
        ).reshape(batch * commands, self.num_candidates, self.model_dim)
        trajectory_tokens = self.trajectory_encoder(sorted_trajectories.flatten(-2))
        tokens = trajectory_tokens + self.context_fusion(
            torch.cat([flat_nav, repeated_sparse, bev_context], dim=-1)
        )
        repeated_boxes = object_boxes[:, None].expand(-1, commands, -1, -1).reshape(
            batch * commands, object_boxes.shape[1], object_boxes.shape[2]
        )
        repeated_visibility = object_visibility[:, None].expand(-1, commands, -1, -1).reshape(
            batch * commands, object_visibility.shape[1], 4
        )
        repeated_class = object_class[:, None].expand(-1, commands, -1).reshape(
            batch * commands, object_class.shape[1]
        )
        repeated_valid = object_valid[:, None].expand(-1, commands, -1).reshape(
            batch * commands, object_valid.shape[1]
        )
        visibility, class_evidence, iou = self.visibility_aggregator(
            sorted_trajectories,
            repeated_boxes,
            repeated_visibility,
            repeated_class,
            repeated_valid,
        )
        mamba_visibility = torch.cat([visibility, class_evidence.to(visibility.dtype)], dim=-1)
        any_evidence = class_evidence.any(dim=-1, keepdim=True)
        mamba_result = self.mamba(
            tokens, mamba_visibility, any_evidence, return_diagnostics=return_diagnostics
        )
        if return_diagnostics:
            refined_tokens, diagnostics = mamba_result
        else:
            refined_tokens = mamba_result
            diagnostics = None
        residual = self.residual_head(refined_tokens).reshape_as(sorted_trajectories)
        sorted_output = sorted_trajectories + residual.cumsum(dim=-2)
        sorted_scores = self.score_head(refined_tokens).squeeze(-1)
        output = apply_candidate_permutation(sorted_output, inverse)
        scores = apply_candidate_permutation(sorted_scores, inverse)
        base_increment = expanded.clone()
        base_increment[..., 1:, :] = expanded[..., 1:, :] - expanded[..., :-1, :]
        output_increment = output.reshape(batch, commands, self.num_candidates, steps, 2)
        output_increment[..., 1:, :] = output_increment[..., 1:, :] - output_increment[..., :-1, :]
        result = {
            "prediction": output_increment,
            "score": scores.reshape(batch, commands, self.num_candidates),
            "base_increment": base_increment,
            "permutation": permutation.reshape(batch, commands, self.num_candidates),
            "inverse_permutation": inverse.reshape(batch, commands, self.num_candidates),
            "candidate_visibility": visibility.reshape(batch, commands, self.num_candidates, 8),
            "evidence_mask": class_evidence.reshape(batch, commands, self.num_candidates, 2),
            "iou": iou.reshape(batch, commands, *iou.shape[1:]),
        }
        if diagnostics is not None:
            result["diagnostics"] = diagnostics
        return result
