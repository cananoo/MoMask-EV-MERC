from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        lora_output = self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling
        return base_output + lora_output


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def inject_lora(
    module: nn.Module,
    target_keywords: Iterable[str] = ("query", "value"),
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    target_keywords = tuple(target_keywords)
    replaced = 0
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.Linear) and any(keyword in child_name for keyword in target_keywords):
            setattr(module, child_name, LoRALinear(child_module, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
            continue
        replaced += inject_lora(
            child_module,
            target_keywords=target_keywords,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    return replaced

