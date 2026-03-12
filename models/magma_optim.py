from __future__ import annotations

from typing import Any

import torch


class Magma:
    def __init__(
        self,
        params,
        base_optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float = 2e-4,
        weight_decay: float = 1e-2,
        magma_beta: float = 0.9,
        magma_mask_prob: float = 0.35,
        magma_momentum_source: str = "decoupled",
        **base_optimizer_kwargs: Any,
    ):
        self.base_optimizer = base_optimizer_cls(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **base_optimizer_kwargs,
        )
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.magma_state: dict[torch.nn.Parameter, dict[str, torch.Tensor]] = {}
        self.magma_beta = magma_beta
        self.magma_mask_prob = magma_mask_prob
        self.magma_momentum_source = magma_momentum_source
        self.last_step_stats = {
            "conflict_ratio": 0.0,
            "masked_ratio": 0.0,
            "layerwise": {},
        }
        self.parameter_names: dict[torch.nn.Parameter, str] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def register_parameter_names(self, named_parameters) -> None:
        self.parameter_names = {parameter: name for name, parameter in named_parameters}

    def _bucket_parameter(self, parameter: torch.nn.Parameter) -> str:
        name = self.parameter_names.get(parameter, "")
        if "lora_" in name:
            return "text_lora"
        if "text_encoder" in name:
            return "text_encoder"
        if any(keyword in name for keyword in ("text_projection", "audio_projection", "visual_projection")):
            return "projection"
        if any(keyword in name for keyword in ("ev_gate", "fallback_fusion")):
            return "fusion"
        if "context_encoder" in name:
            return "context"
        if "classifier" in name:
            return "classifier"
        if "speaker_embedding" in name:
            return "speaker"
        return "other"

    @torch.no_grad()
    def step(self, closure=None):
        total_elements = 0
        total_conflicts = 0
        total_masked = 0
        layerwise_totals: dict[str, dict[str, int]] = {}

        for group in self.param_groups:
            beta = group.get("magma_beta", self.magma_beta)
            mask_prob = group.get("magma_mask_prob", self.magma_mask_prob)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    continue

                magma_state = self.magma_state.setdefault(parameter, {})
                if self.magma_momentum_source == "adamw_expavg":
                    adam_state = self.base_optimizer.state.setdefault(parameter, {})
                    exp_avg = adam_state.get("exp_avg")
                    if exp_avg is None:
                        exp_avg = torch.zeros_like(gradient)
                    beta1 = group.get("betas", self.base_optimizer.defaults.get("betas", (0.9, 0.999)))[0]
                    momentum = exp_avg.detach().clone()
                    momentum.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                else:
                    momentum = magma_state.get("momentum")
                    if momentum is None:
                        momentum = torch.zeros_like(gradient)
                    momentum.mul_(beta).add_(gradient, alpha=1.0 - beta)

                conflict_mask = (gradient * momentum) < 0
                random_mask = torch.rand_like(gradient, dtype=torch.float32) < mask_prob
                magma_mask = conflict_mask & random_mask

                gradient.masked_fill_(magma_mask, 0.0)
                if self.magma_momentum_source == "decoupled":
                    magma_state["momentum"] = momentum

                bucket = self._bucket_parameter(parameter)
                bucket_stats = layerwise_totals.setdefault(
                    bucket,
                    {
                        "elements": 0,
                        "conflicts": 0,
                        "masked": 0,
                    },
                )
                total_elements += gradient.numel()
                total_conflicts += int(conflict_mask.sum().item())
                total_masked += int(magma_mask.sum().item())
                bucket_stats["elements"] += gradient.numel()
                bucket_stats["conflicts"] += int(conflict_mask.sum().item())
                bucket_stats["masked"] += int(magma_mask.sum().item())

        loss = self.base_optimizer.step(closure)
        denominator = max(total_elements, 1)
        layerwise = {
            bucket: {
                "masked_ratio": stats["masked"] / max(stats["elements"], 1),
                "conflict_ratio": stats["conflicts"] / max(stats["elements"], 1),
                "masked_elements": stats["masked"],
                "conflict_elements": stats["conflicts"],
                "total_elements": stats["elements"],
            }
            for bucket, stats in layerwise_totals.items()
        }
        self.last_step_stats = {
            "conflict_ratio": total_conflicts / denominator,
            "masked_ratio": total_masked / denominator,
            "layerwise": layerwise,
        }
        return loss

    def state_dict(self) -> dict[str, Any]:
        serialized_magma_state = {id(parameter): {key: value.cpu() for key, value in state.items()} for parameter, state in self.magma_state.items()}
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "magma_beta": self.magma_beta,
            "magma_mask_prob": self.magma_mask_prob,
            "magma_momentum_source": self.magma_momentum_source,
            "last_step_stats": self.last_step_stats,
            "magma_state": serialized_magma_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.magma_beta = state_dict.get("magma_beta", self.magma_beta)
        self.magma_mask_prob = state_dict.get("magma_mask_prob", self.magma_mask_prob)
        self.magma_momentum_source = state_dict.get("magma_momentum_source", self.magma_momentum_source)
        self.last_step_stats = state_dict.get("last_step_stats", self.last_step_stats)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
