# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.merc_model import MultimodalERCModel
from utils.data import collate_conversations, load_dataset_bundle


def make_loader(bundle: dict) -> DataLoader:
    return DataLoader(bundle['test'], batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_conversations, pin_memory=torch.cuda.is_available())


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def resolve_checkpoint(base_dir: Path, want_adamw: bool) -> Path:
    candidates = sorted(base_dir.glob('*/best.pt'))
    if not candidates:
        raise FileNotFoundError(f'No checkpoint found under {base_dir}')
    matches = [path for path in candidates if ('adamw' in path.parent.name.lower()) == want_adamw]
    if not matches:
        raise FileNotFoundError(f'No matching checkpoint found under {base_dir}')
    return matches[0]


@torch.no_grad()
def infer(model: MultimodalERCModel, batch: dict) -> torch.Tensor:
    return model(
        text=batch['text'],
        audio=batch['audio'],
        visual=batch['visual'],
        speaker_ids=batch['speaker_ids'],
        attention_mask=batch['attention_mask'],
    )['logits']


def load_model(dataset: str, ckpt_path: Path, use_ev_gate: bool) -> tuple[MultimodalERCModel, dict, torch.device]:
    bundle = load_dataset_bundle(dataset, validation_ratio=0.1, seed=42)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hidden_dim = ckpt['model']['text_projection.1.weight'].shape[0]
    model = MultimodalERCModel(
        text_dim=bundle['text_dim'],
        audio_dim=bundle['audio_dim'],
        visual_dim=bundle['visual_dim'],
        hidden_dim=hidden_dim,
        num_classes=bundle['num_classes'],
        num_speakers=bundle['num_speakers'],
        dropout=0.2,
        use_ev_gate=use_ev_gate,
    )
    model.load_state_dict(ckpt['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, bundle, device


def apply_condition(batch: dict, condition: str) -> dict:
    if condition == 'clean':
        return batch
    batch = dict(batch)
    if condition == 'audio_missing':
        batch['audio'] = torch.zeros_like(batch['audio'])
    elif condition == 'visual_missing':
        batch['visual'] = torch.zeros_like(batch['visual'])
    elif condition == 'audio_noise':
        scale = batch['audio'].std().clamp_min(1e-6)
        batch['audio'] = batch['audio'] + 0.5 * scale * torch.randn_like(batch['audio'])
    elif condition == 'visual_noise':
        scale = batch['visual'].std().clamp_min(1e-6)
        batch['visual'] = batch['visual'] + 0.5 * scale * torch.randn_like(batch['visual'])
    else:
        raise ValueError(condition)
    return batch


@torch.no_grad()
def evaluate(dataset: str, ckpt_path: Path, use_ev_gate: bool) -> dict:
    torch.manual_seed(42)
    model, bundle, device = load_model(dataset, ckpt_path, use_ev_gate)
    loader = make_loader(bundle)
    results = {}
    for condition in ['clean', 'audio_missing', 'visual_missing', 'audio_noise', 'visual_noise']:
        y_true, y_pred = [], []
        torch.manual_seed(42)
        for batch in loader:
            batch = move_batch(batch, device)
            batch = apply_condition(batch, condition)
            logits = infer(model, batch)
            pred = logits.argmax(dim=-1)
            mask = batch['labels'] != -100
            y_true.append(batch['labels'][mask].cpu().numpy())
            y_pred.append(pred[mask].cpu().numpy())
        yt = np.concatenate(y_true)
        yp = np.concatenate(y_pred)
        results[condition] = {
            'weighted_f1': float(f1_score(yt, yp, average='weighted')),
            'accuracy': float(accuracy_score(yt, yp)),
        }
    return results


def main() -> None:
    compare_dir = ROOT / 'checkpoints' / 'final_iemocap_compare_h320_seeded'
    out = {
        'iemocap': {
            'evgate_adamw': evaluate('iemocap', resolve_checkpoint(compare_dir, want_adamw=True), True),
            'evgate_momask': evaluate('iemocap', resolve_checkpoint(compare_dir, want_adamw=False), True),
        }
    }
    out_path = ROOT / 'checkpoints' / 'analysis' / 'robustness_results.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
