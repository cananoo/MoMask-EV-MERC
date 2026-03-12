from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.merc_model import MultimodalERCModel
from tools.run_controlled_studies import DATASET_CONFIGS, maybe_run
from utils.data import collate_conversations, load_dataset_bundle



def run_decoupled_ema_ablation() -> dict:
    args = SimpleNamespace(resume=False, max_train_batches=None, max_eval_batches=None, cpu=False)
    root = ROOT / "checkpoints" / "additional_reviewer_analyses" / "ema_ablation"
    results = {}
    for source in ["decoupled", "adamw_expavg"]:
        runs = []
        for fold_id in [1, 2, 3, 4, 5]:
            config = {
                **DATASET_CONFIGS["iemocap"],
                "optimizer": "magma",
                "seed": 42,
                "use_ev_gate": True,
                "ev_gate_type": "scalar",
                "ev_gate_distance": "l2",
                "ev_gate_anchor": "text",
                "magma_beta": 0.9,
                "magma_mask_prob": 0.35,
                "magma_momentum_source": source,
                "protocol": "session_5fold",
                "fold_id": fold_id,
            }
            out = root / source / f"fold{fold_id}"
            runs.append(maybe_run("iemocap", config, out, args))
        weighted = [run["test_weighted_f1"] for run in runs]
        macro = [run["test_macro_f1"] for run in runs]
        acc = [run["test_accuracy"] for run in runs]
        results[source] = {
            "weighted_f1_mean": float(np.mean(weighted)),
            "weighted_f1_std": float(np.std(weighted, ddof=1)),
            "macro_f1_mean": float(np.mean(macro)),
            "accuracy_mean": float(np.mean(acc)),
            "runs": runs,
        }
    return results


def build_model_and_loader(dataset: str, checkpoint_path: Path, run_config: dict) -> tuple[torch.nn.Module, torch.utils.data.DataLoader, list[str], torch.device]:
    bundle = load_dataset_bundle(dataset, validation_ratio=0.1, seed=run_config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalERCModel(
        text_dim=bundle["text_dim"],
        audio_dim=bundle["audio_dim"],
        visual_dim=bundle["visual_dim"],
        hidden_dim=run_config["hidden_dim"],
        num_classes=bundle["num_classes"],
        num_speakers=bundle["num_speakers"],
        dropout=run_config["dropout"],
        use_ev_gate=run_config["use_ev_gate"],
        ev_gate_type=run_config.get("ev_gate_type", "scalar"),
        ev_gate_distance=run_config.get("ev_gate_distance", "l2"),
        ev_gate_anchor=run_config.get("ev_gate_anchor", "text"),
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = torch.utils.data.DataLoader(
        bundle["test"],
        batch_size=4,
        shuffle=False,
        collate_fn=collate_conversations,
    )
    labels = bundle["label_names"]
    return model, loader, labels, device


@torch.no_grad()
def classwise_and_anchor_analysis(dataset: str, summary: dict) -> dict:
    checkpoint_path = ROOT / summary["checkpoint_path"]
    model, loader, labels, device = build_model_and_loader(dataset, checkpoint_path, summary["run_config"])
    y_true, y_pred = [], []
    all_anchor_t, all_anchor_a, all_anchor_v = [], [], []
    class_anchor = {label: {"t": [], "a": [], "v": []} for label in labels}

    for batch in loader:
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(
            text=moved["text"],
            audio=moved["audio"],
            visual=moved["visual"],
            speaker_ids=moved["speaker_ids"],
            attention_mask=moved["attention_mask"],
        )
        preds = outputs["logits"].argmax(dim=-1)
        mask = moved["labels"] != -100
        true_flat = moved["labels"][mask].detach().cpu().numpy()
        pred_flat = preds[mask].detach().cpu().numpy()
        y_true.append(true_flat)
        y_pred.append(pred_flat)

        anchor_t = outputs.get("anchor_text_weight")
        anchor_a = outputs.get("anchor_audio_weight")
        anchor_v = outputs.get("anchor_visual_weight")
        if anchor_t is not None:
            anchor_t = anchor_t[mask].detach().cpu().numpy()
            anchor_a = anchor_a[mask].detach().cpu().numpy()
            anchor_v = anchor_v[mask].detach().cpu().numpy()
            all_anchor_t.extend(anchor_t.tolist())
            all_anchor_a.extend(anchor_a.tolist())
            all_anchor_v.extend(anchor_v.tolist())
            for label_index, label_name in enumerate(labels):
                label_mask = true_flat == label_index
                if np.any(label_mask):
                    class_anchor[label_name]["t"].extend(anchor_t[label_mask].tolist())
                    class_anchor[label_name]["a"].extend(anchor_a[label_mask].tolist())
                    class_anchor[label_name]["v"].extend(anchor_v[label_mask].tolist())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    per_class = f1_score(y_true, y_pred, average=None)
    result = {
        "labels": labels,
        "per_class_f1": {label: float(score) for label, score in zip(labels, per_class)},
    }
    if all_anchor_t:
        result["anchor_weight_mean"] = {
            "text": float(np.mean(all_anchor_t)),
            "audio": float(np.mean(all_anchor_a)),
            "visual": float(np.mean(all_anchor_v)),
        }
        result["anchor_weight_by_class"] = {
            label: {
                key: float(np.mean(values)) if values else 0.0
                for key, values in stats.items()
            }
            for label, stats in class_anchor.items()
        }
    return result


def main() -> None:
    out_root = ROOT / "checkpoints" / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    ema = run_decoupled_ema_ablation()
    (out_root / "ema_ablation_summary.json").write_text(json.dumps(ema, indent=2), encoding="utf-8")

    daa_summary = json.loads((ROOT / "checkpoints" / "daa_revision_single_runs" / "summary.json").read_text(encoding="utf-8"))
    daa_iemocap = classwise_and_anchor_analysis("iemocap", daa_summary["iemocap"])
    (out_root / "daa_anchor_classwise_iemocap.json").write_text(json.dumps(daa_iemocap, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
