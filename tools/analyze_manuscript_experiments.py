# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.merc_model import MultimodalERCModel
from utils.data import collate_conversations, load_dataset_bundle
FIG_DIR = ROOT / 'figures'
OUT_DIR = ROOT / 'checkpoints' / 'analysis'
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'no-latex', 'grid'])
except Exception:
    sns.set_theme(style='whitegrid', context='talk')


def make_loader(bundle: dict, batch_size: int = 4) -> DataLoader:
    return DataLoader(
        bundle['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_conversations,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def forward_with_hidden(model: MultimodalERCModel, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    text_hidden = model.text_projection(batch['text'])
    audio_hidden = model.audio_projection(batch['audio'])
    visual_hidden = model.visual_projection(batch['visual'])
    speaker_hidden = model.speaker_embedding(batch['speaker_ids'])
    if model.use_ev_gate:
        fused_hidden, _ = model.ev_gate(text_hidden, audio_hidden, visual_hidden, batch['attention_mask'])
    else:
        fused_hidden = model.fallback_fusion(torch.cat([text_hidden, audio_hidden, visual_hidden], dim=-1))
    fused_hidden = fused_hidden + speaker_hidden
    context_hidden, _ = model.context_encoder(fused_hidden)
    logits = model.classifier(context_hidden)
    return logits, context_hidden


@torch.no_grad()
def collect_predictions(dataset: str, checkpoint_path: Path, use_ev_gate: bool, batch_size: int = 4) -> dict:
    bundle = load_dataset_bundle(dataset, validation_ratio=0.1, seed=42)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
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

    loader = make_loader(bundle, batch_size=batch_size)
    all_true, all_pred, all_conv, all_embed = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        logits, context_hidden = forward_with_hidden(model, batch)
        preds = logits.argmax(dim=-1)
        mask = batch['labels'] != -100
        all_true.append(batch['labels'][mask].cpu().numpy())
        all_pred.append(preds[mask].cpu().numpy())
        all_embed.append(context_hidden[mask].cpu().numpy())
        lengths = batch['lengths'].cpu().tolist()
        for conv_id, length in zip(batch['conversation_ids'], lengths):
            all_conv.extend([conv_id] * int(length))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    embeddings = np.concatenate(all_embed)
    return {
        'dataset': dataset,
        'checkpoint_path': str(checkpoint_path),
        'use_ev_gate': use_ev_gate,
        'label_names': bundle['label_names'],
        'y_true': y_true,
        'y_pred': y_pred,
        'conversation_ids': np.asarray(all_conv),
        'embeddings': embeddings,
    }


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if metric == 'weighted_f1':
            scores.append(f1_score(yt, yp, average='weighted'))
        elif metric == 'accuracy':
            scores.append(accuracy_score(yt, yp))
        else:
            raise ValueError(metric)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def per_conversation_weighted_f1(result: dict) -> dict[str, float]:
    grouped_true = defaultdict(list)
    grouped_pred = defaultdict(list)
    for conv, yt, yp in zip(result['conversation_ids'], result['y_true'], result['y_pred']):
        grouped_true[conv].append(int(yt))
        grouped_pred[conv].append(int(yp))
    scores = {}
    for conv in grouped_true:
        scores[conv] = float(f1_score(grouped_true[conv], grouped_pred[conv], average='weighted'))
    return scores


def mcnemar_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[int, int, float]:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b_only = int(np.sum(a_correct & ~b_correct))
    c_only = int(np.sum(~a_correct & b_correct))
    total = b_only + c_only
    pvalue = 1.0 if total == 0 else float(binomtest(min(b_only, c_only), total, 0.5, alternative='two-sided').pvalue)
    return b_only, c_only, pvalue


def cohens_dz(conv_scores_a: dict[str, float], conv_scores_b: dict[str, float]) -> float:
    keys = sorted(set(conv_scores_a) & set(conv_scores_b))
    diffs = np.asarray([conv_scores_b[k] - conv_scores_a[k] for k in keys], dtype=np.float64)
    if len(diffs) < 2:
        return 0.0
    std = diffs.std(ddof=1)
    if std == 0:
        return 0.0
    return float(diffs.mean() / std)


def count_params(dataset: str, use_ev_gate: bool, hidden_dim: int = 320) -> float:
    bundle = load_dataset_bundle(dataset, validation_ratio=0.1, seed=42)
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
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def measure_latency(dataset: str, checkpoint_path: Path, use_ev_gate: bool, repeats: int = 24) -> tuple[float, float]:
    bundle = load_dataset_bundle(dataset, validation_ratio=0.1, seed=42)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
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
    loader = make_loader(bundle, batch_size=1)
    batches = []
    for i, batch in enumerate(loader):
        batches.append(move_batch(batch, device))
        if len(batches) >= repeats:
            break
    if not batches:
        return 0.0, 0.0
    for batch in batches[:5]:
        _ = forward_with_hidden(model, batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times = []
    for batch in batches:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = forward_with_hidden(model, batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(times)), float(np.std(times, ddof=1) if len(times) > 1 else 0.0)


def resolve_checkpoint(base_dir: Path, want_adamw: bool) -> Path:
    candidates = sorted(base_dir.glob("*/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {base_dir}")
    matches = [path for path in candidates if ("adamw" in path.parent.name.lower()) == want_adamw]
    if not matches:
        raise FileNotFoundError(f"No matching checkpoint found under {base_dir}")
    return matches[0]


def read_summary_entry(summary_path: Path, want_adamw: bool) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if want_adamw:
        return summary["adamw"]
    non_adamw_keys = [key for key in summary if key != "adamw"]
    if not non_adamw_keys:
        raise KeyError(f"No non-AdamW entry found in {summary_path}")
    return summary[non_adamw_keys[0]]


def resolve_experiment_dir(pattern: str, want_adamw: bool) -> Path:
    candidates = sorted((ROOT / "checkpoints").glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No experiment directory matched {pattern}")
    matches = [path for path in candidates if ("adamw" in path.name.lower()) == want_adamw]
    if not matches:
        raise FileNotFoundError(f"No matching experiment directory found for {pattern}")
    return matches[0]


def plot_tsne(baseline: dict, ours: dict, output_prefix: Path) -> None:
    label_names = baseline['label_names']
    rng = np.random.default_rng(42)
    sample_size = min(900, len(baseline['y_true']), len(ours['y_true']))
    baseline_idx = rng.choice(len(baseline['y_true']), size=sample_size, replace=False)
    ours_idx = rng.choice(len(ours['y_true']), size=sample_size, replace=False)

    def project(embeddings: np.ndarray) -> np.ndarray:
        pca_dim = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
        reduced = PCA(n_components=pca_dim, random_state=42).fit_transform(embeddings) if pca_dim >= 2 else embeddings
        perplexity = min(35, max(5, embeddings.shape[0] // 20))
        return TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perplexity, random_state=42).fit_transform(reduced)

    baseline_xy = project(baseline['embeddings'][baseline_idx])
    ours_xy = project(ours['embeddings'][ours_idx])
    baseline_labels = baseline['y_true'][baseline_idx]
    ours_labels = ours['y_true'][ours_idx]

    palette = sns.color_palette('tab10', n_colors=len(label_names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
    titles = ['EV-Gate + AdamW', 'EV-Gate + MoMask']
    for ax, xy, labels, title in zip(axes, [baseline_xy, ours_xy], [baseline_labels, ours_labels], titles):
        for label_id, label_name in enumerate(label_names):
            mask = labels == label_id
            if mask.sum() == 0:
                continue
            ax.scatter(xy[mask, 0], xy[mask, 1], s=16, alpha=0.75, color=palette[label_id], label=label_name, edgecolors='none')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('t-SNE-1')
        ax.set_ylabel('t-SNE-2')
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=8, markerfacecolor=palette[i], markeredgecolor='none', label=name) for i, name in enumerate(label_names)]
    fig.legend(handles=handles, labels=label_names, loc='lower center', ncol=min(6, len(label_names)), frameon=False)
    fig.suptitle('IEMOCAP utterance representations after contextual encoding', fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    iemocap_compare_dir = ROOT / 'checkpoints' / 'final_iemocap_compare_h320_seeded'
    iemocap_noev_dir = resolve_experiment_dir('final_iemocap_noev*_h320', want_adamw=False)
    meld_compare_dir = ROOT / 'checkpoints' / 'final_meld_compare_tuned'
    paths = {
        'iemocap_adamw': resolve_checkpoint(iemocap_compare_dir, want_adamw=True),
        'iemocap_momask': resolve_checkpoint(iemocap_compare_dir, want_adamw=False),
        'iemocap_noev_momask': resolve_checkpoint(iemocap_noev_dir, want_adamw=False),
        'iemocap_noev_adamw': resolve_checkpoint(ROOT / 'checkpoints' / 'final_iemocap_noev_adamw_h320', want_adamw=True),
        'meld_adamw': resolve_checkpoint(meld_compare_dir, want_adamw=True),
        'meld_momask': resolve_checkpoint(meld_compare_dir, want_adamw=False),
    }

    iemocap_adamw = collect_predictions('iemocap', paths['iemocap_adamw'], use_ev_gate=True)
    iemocap_momask = collect_predictions('iemocap', paths['iemocap_momask'], use_ev_gate=True)
    meld_adamw = collect_predictions('meld', paths['meld_adamw'], use_ev_gate=True)
    meld_momask = collect_predictions('meld', paths['meld_momask'], use_ev_gate=True)

    stats = {}
    for name, result in {
        'iemocap_adamw': iemocap_adamw,
        'iemocap_momask': iemocap_momask,
        'meld_adamw': meld_adamw,
        'meld_momask': meld_momask,
    }.items():
        wf1 = f1_score(result['y_true'], result['y_pred'], average='weighted')
        acc = accuracy_score(result['y_true'], result['y_pred'])
        wf1_ci = bootstrap_metric(result['y_true'], result['y_pred'], 'weighted_f1')
        acc_ci = bootstrap_metric(result['y_true'], result['y_pred'], 'accuracy')
        stats[name] = {
            'weighted_f1': float(wf1),
            'weighted_f1_ci': [float(wf1_ci[0]), float(wf1_ci[1])],
            'accuracy': float(acc),
            'accuracy_ci': [float(acc_ci[0]), float(acc_ci[1])],
        }

    for dataset, base_result, ours_result in [
        ('iemocap', iemocap_adamw, iemocap_momask),
        ('meld', meld_adamw, meld_momask),
    ]:
        b_only, c_only, pvalue = mcnemar_pvalue(base_result['y_true'], base_result['y_pred'], ours_result['y_pred'])
        d_value = cohens_dz(per_conversation_weighted_f1(base_result), per_conversation_weighted_f1(ours_result))
        stats[f'{dataset}_comparison'] = {
            'mcnemar_b_only': b_only,
            'mcnemar_c_only': c_only,
            'mcnemar_pvalue': float(pvalue),
            'cohens_dz_conversation_f1': float(d_value),
        }

    ablation = {
        'Base (Linear fusion + AdamW)': read_summary_entry(ROOT / 'checkpoints' / 'final_iemocap_noev_adamw_h320' / 'summary.json', want_adamw=True),
        'Base + EV-Gate': read_summary_entry(iemocap_compare_dir / 'summary.json', want_adamw=True),
        'Base + MoMask': read_summary_entry(iemocap_noev_dir / 'summary.json', want_adamw=False),
        'Ours (EV-Gate + MoMask)': read_summary_entry(iemocap_compare_dir / 'summary.json', want_adamw=False),
    }

    complexity = {}
    complexity_configs = {
        'Base (Linear fusion + AdamW)': (paths['iemocap_noev_adamw'], False),
        'Base + EV-Gate': (paths['iemocap_adamw'], True),
        'Base + MoMask': (paths['iemocap_noev_momask'], False),
        'Ours (EV-Gate + MoMask)': (paths['iemocap_momask'], True),
    }
    for name, (ckpt_path, use_ev_gate) in complexity_configs.items():
        params_m = count_params('iemocap', use_ev_gate=use_ev_gate, hidden_dim=320)
        latency_ms, latency_std = measure_latency('iemocap', ckpt_path, use_ev_gate=use_ev_gate)
        complexity[name] = {
            'params_m': float(params_m),
            'latency_ms_per_dialogue': float(latency_ms),
            'latency_std_ms': float(latency_std),
        }

    plot_tsne(iemocap_adamw, iemocap_momask, FIG_DIR / 'iemocap_tsne_comparison')

    payload = {
        'stats': stats,
        'ablation': ablation,
        'complexity': complexity,
    }
    (OUT_DIR / 'manuscript_analysis.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

