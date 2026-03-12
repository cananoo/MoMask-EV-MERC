from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConflictStats:
    cosine_conflict: float = 0.0
    grad_norm: float = 0.0


def _flatten_grad_list(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for grad, parameter in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros_like(parameter, memory_format=torch.preserve_format).reshape(-1))
        else:
            chunks.append(grad.detach().reshape(-1))
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


def _simplex_projection(weights: torch.Tensor) -> torch.Tensor:
    if weights.numel() == 1:
        return torch.ones_like(weights)
    sorted_weights, _ = torch.sort(weights, descending=True)
    cssv = torch.cumsum(sorted_weights, dim=0) - 1
    indices = torch.arange(1, weights.numel() + 1, device=weights.device, dtype=weights.dtype)
    cond = sorted_weights - cssv / indices > 0
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
    theta = cssv[rho] / float(rho + 1)
    return torch.clamp(weights - theta, min=0.0)


class ConflictOptimizer:
    def __init__(
        self,
        params,
        method: str,
        lr: float,
        weight_decay: float,
        cagrad_c: float = 0.4,
    ):
        if method not in {"pcgrad", "cagrad", "mgda"}:
            raise ValueError(f"Unsupported conflict optimizer: {method}")
        self.method = method
        self.base_optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.cagrad_c = cagrad_c
        self.last_step_stats = {
            "conflict_ratio": 0.0,
            "masked_ratio": 0.0,
            "combined_grad_norm": 0.0,
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "method": self.method,
            "cagrad_c": self.cagrad_c,
            "last_step_stats": self.last_step_stats,
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.method = state_dict.get("method", self.method)
        self.cagrad_c = state_dict.get("cagrad_c", self.cagrad_c)
        self.last_step_stats = state_dict.get("last_step_stats", self.last_step_stats)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def step(self, closure=None):
        return self.base_optimizer.step(closure)

    def _pairwise_conflict_ratio(self, grads: torch.Tensor) -> float:
        if grads.shape[0] <= 1:
            return 0.0
        conflicts = 0
        pairs = 0
        for i in range(grads.shape[0]):
            for j in range(i + 1, grads.shape[0]):
                denom = grads[i].norm() * grads[j].norm() + 1e-8
                cosine = torch.dot(grads[i], grads[j]) / denom
                conflicts += int(cosine.item() < 0.0)
                pairs += 1
        return conflicts / max(pairs, 1)

    def combine(self, task_grads: list[tuple[torch.Tensor | None, ...]], params: list[torch.nn.Parameter]) -> tuple[torch.Tensor, ConflictStats]:
        grad_matrix = torch.stack([_flatten_grad_list(grad_tuple, params) for grad_tuple in task_grads], dim=0)
        stats = ConflictStats(cosine_conflict=self._pairwise_conflict_ratio(grad_matrix))
        if grad_matrix.numel() == 0:
            return grad_matrix.sum(dim=0), stats
        if self.method == "pcgrad":
            combined = self._pcgrad(grad_matrix)
        elif self.method == "cagrad":
            combined = self._cagrad(grad_matrix)
        else:
            combined = self._mgda(grad_matrix)
        stats.grad_norm = float(combined.norm().item())
        return combined, stats

    def _pcgrad(self, grad_matrix: torch.Tensor) -> torch.Tensor:
        projected = grad_matrix.clone()
        order = list(range(projected.shape[0]))
        for i in range(projected.shape[0]):
            gi = projected[i]
            for j in order:
                if i == j:
                    continue
                gj = projected[j]
                denom = gj.dot(gj) + 1e-12
                inner = gi.dot(gj)
                if inner < 0:
                    gi = gi - inner / denom * gj
            projected[i] = gi
        return projected.mean(dim=0)

    def _mgda(self, grad_matrix: torch.Tensor) -> torch.Tensor:
        num_tasks = grad_matrix.shape[0]
        if num_tasks == 1:
            return grad_matrix[0]
        gram = grad_matrix @ grad_matrix.t()
        weights = torch.full((num_tasks,), 1.0 / num_tasks, device=grad_matrix.device, dtype=grad_matrix.dtype)
        step_size = 0.1
        for _ in range(80):
            grad_w = 2.0 * gram @ weights
            weights = _simplex_projection(weights - step_size * grad_w)
        return (weights.unsqueeze(1) * grad_matrix).sum(dim=0)

    def _cagrad(self, grad_matrix: torch.Tensor) -> torch.Tensor:
        num_tasks = grad_matrix.shape[0]
        if num_tasks == 1:
            return grad_matrix[0]
        gram = grad_matrix @ grad_matrix.t()
        avg_weights = torch.full((num_tasks,), 1.0 / num_tasks, device=grad_matrix.device, dtype=grad_matrix.dtype)
        weights = avg_weights.clone()
        step_size = 0.1
        mean_term = gram.mean(dim=1)
        for _ in range(80):
            quad = torch.sqrt(torch.clamp(weights @ gram @ weights, min=1e-12))
            grad_w = mean_term + self.cagrad_c * (gram @ weights) / quad
            weights = _simplex_projection(weights - step_size * grad_w)
        gw = (weights.unsqueeze(1) * grad_matrix).sum(dim=0)
        g0 = grad_matrix.mean(dim=0)
        correction = self.cagrad_c * gw / (gw.norm() + 1e-12)
        return g0 + correction
