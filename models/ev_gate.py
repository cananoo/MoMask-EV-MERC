from __future__ import annotations

import torch
from torch import nn


class EVGate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.2,
        gate_type: str = "scalar",
        distance_type: str = "l2",
        anchor_mode: str = "text",
    ):
        super().__init__()
        if gate_type not in {"scalar", "vector"}:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
        if distance_type not in {"l2", "cosine"}:
            raise ValueError(f"Unsupported distance_type: {distance_type}")
        if anchor_mode not in {"text", "mean", "learned"}:
            raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")

        self.hidden_dim = hidden_dim
        self.gate_type = gate_type
        self.distance_type = distance_type
        self.anchor_mode = anchor_mode
        self.route_input_dim = 2 if anchor_mode == "text" else 3
        self.distance_norm = nn.LayerNorm(self.route_input_dim)
        self.stable_path = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        conflict_input_multiplier = 6
        self.conflict_path = nn.Sequential(
            nn.Linear(hidden_dim * conflict_input_multiplier, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        router_hidden = max(hidden_dim // 2, 8)
        router_output_dim = 1 if gate_type == "scalar" else hidden_dim
        self.scalar_router = nn.Sequential(
            nn.Linear(self.route_input_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, router_output_dim),
        )
        self.vector_router_norm = nn.LayerNorm(hidden_dim * 3)
        self.vector_router = nn.Linear(hidden_dim * 3, hidden_dim)
        self.anchor_scorer = None
        if anchor_mode == "learned":
            self.anchor_scorer = nn.Sequential(
                nn.Linear(hidden_dim, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, 1),
            )
        self.conflict_scale = nn.Parameter(torch.tensor(1.0))
        self.violation_bias = nn.Parameter(torch.tensor(0.0))
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _distance(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.distance_type == "l2":
            return torch.norm(left - right, dim=-1, keepdim=True)
        cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1, eps=1e-8)
        return (1.0 - cosine).unsqueeze(-1)

    def _anchor(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.anchor_mode == "text":
            return text, {}
        if self.anchor_mode == "mean":
            return (text + audio + visual) / 3.0, {}
        logits = torch.cat(
            [
                self.anchor_scorer(text),
                self.anchor_scorer(audio),
                self.anchor_scorer(visual),
            ],
            dim=-1,
        )
        weights = torch.softmax(logits, dim=-1)
        anchor = (
            weights[..., 0:1] * text
            + weights[..., 1:2] * audio
            + weights[..., 2:3] * visual
        )
        return anchor, {
            "anchor_weight_text": weights[..., 0],
            "anchor_weight_audio": weights[..., 1],
            "anchor_weight_visual": weights[..., 2],
        }

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        anchor_features, anchor_diagnostics = self._anchor(text_features, audio_features, visual_features)
        delta_text = text_features - anchor_features
        delta_audio = audio_features - anchor_features
        delta_visual = visual_features - anchor_features
        text_audio_distance = torch.norm(delta_text - delta_audio, dim=-1, keepdim=True)
        text_visual_distance = torch.norm(delta_text - delta_visual, dim=-1, keepdim=True)
        if self.anchor_mode == "text":
            routed_distances = torch.cat(
                [
                    self._distance(text_features, audio_features),
                    self._distance(text_features, visual_features),
                ],
                dim=-1,
            )
        else:
            routed_distances = torch.cat(
                [
                    self._distance(anchor_features, text_features),
                    self._distance(anchor_features, audio_features),
                    self._distance(anchor_features, visual_features),
                ],
                dim=-1,
            )
        normalized_distances = self.distance_norm(routed_distances)
        gate_input = torch.cat([delta_text, delta_audio, delta_visual], dim=-1)
        if self.gate_type == "scalar":
            violation_gate = torch.sigmoid(
                self.conflict_scale * self.scalar_router(normalized_distances) + self.violation_bias
            )
        else:
            violation_gate = torch.sigmoid(
                self.conflict_scale * self.vector_router(self.vector_router_norm(gate_input)) + self.violation_bias
            )

        conflict_inputs = [
            text_features,
            audio_features,
            visual_features,
            torch.abs(delta_text),
            torch.abs(delta_audio),
            torch.abs(delta_visual),
        ]

        stable_features = self.stable_path(torch.cat([text_features, audio_features, visual_features], dim=-1))
        conflict_features = self.conflict_path(torch.cat(conflict_inputs, dim=-1))
        fused_features = stable_features + violation_gate * conflict_features
        if attention_mask is not None:
            fused_features = fused_features * attention_mask.unsqueeze(-1)

        fused_features = self.output_norm(fused_features)
        diagnostics = {
            "violation_gate": violation_gate.mean(dim=-1) if violation_gate.shape[-1] > 1 else violation_gate.squeeze(-1),
            "anchor_text_weight": anchor_diagnostics.get("anchor_weight_text"),
            "anchor_audio_weight": anchor_diagnostics.get("anchor_weight_audio"),
            "anchor_visual_weight": anchor_diagnostics.get("anchor_weight_visual"),
            "text_audio_distance": text_audio_distance.squeeze(-1),
            "text_visual_distance": text_visual_distance.squeeze(-1),
            "gate_type": self.gate_type,
            "distance_type": self.distance_type,
            "anchor_mode": self.anchor_mode,
            **anchor_diagnostics,
        }
        return fused_features, diagnostics
