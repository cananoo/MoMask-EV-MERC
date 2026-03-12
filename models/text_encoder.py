from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.lora import freeze_module, inject_lora

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:  # pragma: no cover - import guard for minimal environments
    AutoModel = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class RobertaLoRAEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-large",
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        max_length: int = 64,
        pooling: str = "cls",
        target_modules: tuple[str, ...] = ("query", "value"),
        train_layer_norm: bool = False,
    ):
        super().__init__()
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is required for LoRA text finetuning."
            ) from _TRANSFORMERS_IMPORT_ERROR
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.encoder.config.hidden_size)

        freeze_module(self.encoder)
        self.lora_module_count = inject_lora(
            self.encoder,
            target_keywords=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if train_layer_norm:
            for name, parameter in self.encoder.named_parameters():
                if "LayerNorm" in name:
                    parameter.requires_grad = True

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]

        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return summed / denominator

    def forward(
        self,
        utterances: list[list[str]],
        device: torch.device,
        max_seq_len: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size = len(utterances)
        flat_utterances = [utterance for conversation in utterances for utterance in conversation]
        if not flat_utterances:
            empty = torch.zeros(batch_size, max_seq_len, self.hidden_size, device=device)
            return empty, {"lora_module_count": self.lora_module_count}

        tokenized = self.tokenizer(
            flat_utterances,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        outputs = self.encoder(**tokenized)
        utterance_embeddings = self._pool(outputs.last_hidden_state, tokenized["attention_mask"])

        contextualized = torch.zeros(
            batch_size,
            max_seq_len,
            self.hidden_size,
            device=device,
            dtype=utterance_embeddings.dtype,
        )
        cursor = 0
        for batch_index, conversation in enumerate(utterances):
            conversation_length = len(conversation)
            if conversation_length == 0:
                continue
            contextualized[batch_index, :conversation_length] = utterance_embeddings[cursor : cursor + conversation_length]
            cursor += conversation_length

        return contextualized, {
            "lora_module_count": self.lora_module_count,
            "text_encoder_model": self.model_name,
        }

