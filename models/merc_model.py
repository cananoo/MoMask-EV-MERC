from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.text_encoder import RobertaLoRAEncoder


class MultimodalERCModel(nn.Module):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_speakers: int,
        dropout: float = 0.2,
        use_ev_gate: bool = True,
        ev_gate_type: str = "scalar",
        ev_gate_distance: str = "l2",
        ev_gate_anchor: str = "text",
        text_encoder_mode: str = "offline",
        text_encoder_name: str = "roberta-large",
        text_lora_rank: int = 8,
        text_lora_alpha: int = 16,
        text_lora_dropout: float = 0.05,
        text_max_length: int = 64,
        text_pooling: str = "cls",
        text_train_layer_norm: bool = False,
        gate_position: str = "pre_context",
    ):
        super().__init__()
        del num_speakers

        if text_encoder_mode not in {"offline", "lora"}:
            raise ValueError(f"Unsupported text_encoder_mode: {text_encoder_mode}")
        if gate_position not in {"pre_context", "post_context"}:
            raise ValueError(f"Unsupported gate_position: {gate_position}")
        if ev_gate_type not in {"scalar", "vector"}:
            raise ValueError(f"Unsupported ev_gate_type: {ev_gate_type}")
        if ev_gate_distance not in {"l2", "cosine"}:
            raise ValueError(f"Unsupported ev_gate_distance: {ev_gate_distance}")
        if ev_gate_anchor not in {"text", "mean", "learned"}:
            raise ValueError(f"Unsupported ev_gate_anchor: {ev_gate_anchor}")

        self.use_ev_gate = use_ev_gate
        self.ev_gate_type = ev_gate_type
        self.ev_gate_distance = ev_gate_distance
        self.ev_gate_anchor = ev_gate_anchor
        self.text_encoder_mode = text_encoder_mode
        self.gate_position = gate_position
        self.hidden_dim = hidden_dim

        self.text_encoder = None
        effective_text_dim = text_dim
        if text_encoder_mode == "lora":
            self.text_encoder = RobertaLoRAEncoder(
                model_name=text_encoder_name,
                rank=text_lora_rank,
                alpha=text_lora_alpha,
                dropout=text_lora_dropout,
                max_length=text_max_length,
                pooling=text_pooling,
                train_layer_norm=text_train_layer_norm,
            )
            effective_text_dim = self.text_encoder.hidden_size

        self.text_projection = nn.Sequential(
            nn.LayerNorm(effective_text_dim),
            nn.Linear(effective_text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        context_hidden = hidden_dim // 2
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=context_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.bigru_t = None
        self.bigru_a = None
        self.bigru_v = None
        if gate_position == "post_context":
            self.bigru_t = nn.GRU(
                input_size=hidden_dim,
                hidden_size=context_hidden,
                batch_first=True,
                bidirectional=True,
            )
            self.bigru_a = nn.GRU(
                input_size=hidden_dim,
                hidden_size=context_hidden,
                batch_first=True,
                bidirectional=True,
            )
            self.bigru_v = nn.GRU(
                input_size=hidden_dim,
                hidden_size=context_hidden,
                batch_first=True,
                bidirectional=True,
            )

        attn_hidden = max(hidden_dim // 2, 8)
        self.anchor_attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, 1),
        )

        self.stable_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.conflict_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.vector_gate_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.scalar_gate_proj = nn.Sequential(
            nn.Linear(3, max(hidden_dim // 8, 8)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 8, 8), 1),
        )

        self.fallback_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode_text(
        self,
        text: torch.Tensor,
        utterances: list[list[str]] | None,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | str]]:
        diagnostics: dict[str, torch.Tensor | int | str] = {}
        if self.text_encoder_mode == "lora":
            if utterances is None:
                raise ValueError("Utterances are required when text_encoder_mode='lora'.")
            text_hidden, text_diagnostics = self.text_encoder(
                utterances=utterances,
                device=text.device,
                max_seq_len=text.shape[1],
            )
            text_hidden = text_hidden * attention_mask.unsqueeze(-1)
            diagnostics.update(text_diagnostics)
            return text_hidden, diagnostics
        return text, diagnostics

    def _distance(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.ev_gate_distance == "cosine":
            cosine = F.cosine_similarity(left, right, dim=-1, eps=1e-8)
            return (1.0 - cosine).unsqueeze(-1)
        return torch.norm(left - right, dim=-1, keepdim=True)

    def _anchor_weights(
        self,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = torch.cat(
            [self.anchor_attn(h_t), self.anchor_attn(h_a), self.anchor_attn(h_v)],
            dim=-1,
        )
        weights = F.softmax(scores, dim=-1)
        return weights[..., 0:1], weights[..., 1:2], weights[..., 2:3]

    def _resolve_anchor(
        self,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.ev_gate_anchor == "text":
            zeros = torch.zeros_like(h_t[..., :1])
            ones = torch.ones_like(h_t[..., :1])
            return h_t, {
                "anchor_text_weight": ones.squeeze(-1),
                "anchor_audio_weight": zeros.squeeze(-1),
                "anchor_visual_weight": zeros.squeeze(-1),
            }
        if self.ev_gate_anchor == "mean":
            weight = torch.full_like(h_t[..., :1], 1.0 / 3.0)
            anchor = (h_t + h_a + h_v) / 3.0
            return anchor, {
                "anchor_text_weight": weight.squeeze(-1),
                "anchor_audio_weight": weight.squeeze(-1),
                "anchor_visual_weight": weight.squeeze(-1),
            }

        alpha_t, alpha_a, alpha_v = self._anchor_weights(h_t, h_a, h_v)
        anchor = alpha_t * h_t + alpha_a * h_a + alpha_v * h_v
        return anchor, {
            "anchor_text_weight": alpha_t.squeeze(-1),
            "anchor_audio_weight": alpha_a.squeeze(-1),
            "anchor_visual_weight": alpha_v.squeeze(-1),
        }

    def _pre_context_fusion(
        self,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stable_input = torch.cat([h_t, h_a, h_v], dim=-1)
        stable_hidden = self.stable_mlp(stable_input)

        if not self.use_ev_gate:
            fused_hidden = self.fallback_fusion(stable_input)
            diagnostics = {
                "violation_gate": torch.zeros_like(attention_mask, dtype=fused_hidden.dtype),
                "text_audio_distance": torch.norm(h_t - h_a, dim=-1),
                "text_visual_distance": torch.norm(h_t - h_v, dim=-1),
            }
            return fused_hidden, diagnostics

        anchor, anchor_stats = self._resolve_anchor(h_t, h_a, h_v)
        delta_t = h_t - anchor
        delta_a = h_a - anchor
        delta_v = h_v - anchor

        conflict_input = torch.cat(
            [h_t, h_a, h_v, torch.abs(delta_t), torch.abs(delta_a), torch.abs(delta_v)],
            dim=-1,
        )
        conflict_hidden = self.conflict_mlp(conflict_input)

        if self.ev_gate_type == "vector":
            gate_input = torch.cat([delta_t, delta_a, delta_v], dim=-1)
            gate = torch.sigmoid(self.vector_gate_proj(gate_input))
            gate_diag = gate.mean(dim=-1)
        else:
            if self.ev_gate_anchor == "text":
                scalar_features = torch.cat(
                    [self._distance(h_t, h_a), self._distance(h_t, h_v), self._distance(h_a, h_v)],
                    dim=-1,
                )
            else:
                scalar_features = torch.cat(
                    [self._distance(h_t, anchor), self._distance(h_a, anchor), self._distance(h_v, anchor)],
                    dim=-1,
                )
            gate = torch.sigmoid(self.scalar_gate_proj(scalar_features))
            gate_diag = gate.squeeze(-1)

        fused_hidden = stable_hidden + gate * conflict_hidden
        diagnostics = {
            **anchor_stats,
            "violation_gate": gate_diag,
            "text_audio_distance": torch.norm(h_t - h_a, dim=-1),
            "text_visual_distance": torch.norm(h_t - h_v, dim=-1),
        }
        return fused_hidden, diagnostics

    def _post_context_fusion(
        self,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        h_v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.bigru_t is None or self.bigru_a is None or self.bigru_v is None:
            raise RuntimeError("Post-context fusion requested without modality-specific context encoders.")

        u_t, _ = self.bigru_t(h_t)
        u_a, _ = self.bigru_a(h_a)
        u_v, _ = self.bigru_v(h_v)

        if not self.use_ev_gate:
            fused_hidden = self.fallback_fusion(torch.cat([u_t, u_a, u_v], dim=-1))
            diagnostics = {
                "violation_gate": torch.zeros_like(attention_mask, dtype=fused_hidden.dtype),
                "text_audio_distance": torch.norm(u_t - u_a, dim=-1),
                "text_visual_distance": torch.norm(u_t - u_v, dim=-1),
            }
            return fused_hidden, diagnostics

        alpha_t, alpha_a, alpha_v = self._anchor_weights(u_t, u_a, u_v)
        anchor = alpha_t * u_t + alpha_a * u_a + alpha_v * u_v
        delta_t = u_t - anchor
        delta_a = u_a - anchor
        delta_v = u_v - anchor

        stable_hidden = self.stable_mlp(torch.cat([u_t, u_a, u_v], dim=-1))
        conflict_hidden = self.conflict_mlp(
            torch.cat(
                [u_t, u_a, u_v, torch.abs(delta_t), torch.abs(delta_a), torch.abs(delta_v)],
                dim=-1,
            )
        )
        gate = torch.sigmoid(self.vector_gate_proj(torch.cat([delta_t, delta_a, delta_v], dim=-1)))
        fused_hidden = stable_hidden + gate * conflict_hidden
        diagnostics = {
            "anchor_text_weight": alpha_t.squeeze(-1),
            "anchor_audio_weight": alpha_a.squeeze(-1),
            "anchor_visual_weight": alpha_v.squeeze(-1),
            "violation_gate": gate.mean(dim=-1),
            "text_audio_distance": torch.norm(u_t - u_a, dim=-1),
            "text_visual_distance": torch.norm(u_t - u_v, dim=-1),
        }
        return fused_hidden, diagnostics

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        speaker_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        utterances: list[list[str]] | None = None,
    ) -> dict[str, torch.Tensor]:
        del speaker_ids

        raw_text_hidden, text_diagnostics = self._encode_text(text, utterances, attention_mask)
        h_t = self.text_projection(raw_text_hidden)
        h_a = self.audio_projection(audio)
        h_v = self.visual_projection(visual)

        if self.gate_position == "post_context":
            fused_hidden, diagnostics = self._post_context_fusion(h_t, h_a, h_v, attention_mask)
            contextual_hidden = fused_hidden
        else:
            fused_hidden, diagnostics = self._pre_context_fusion(h_t, h_a, h_v, attention_mask)
            contextual_hidden, _ = self.bigru(fused_hidden)

        contextual_hidden = self.output_norm(contextual_hidden)
        contextual_hidden = contextual_hidden * attention_mask.unsqueeze(-1)
        logits = self.classifier(contextual_hidden)

        return {
            "logits": logits,
            "gate_position": self.gate_position,
            **text_diagnostics,
            **diagnostics,
        }
