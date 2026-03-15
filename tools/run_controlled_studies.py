from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.conflict_optim import ConflictOptimizer
from models.momask_optim import MoMask
from models.merc_model import MultimodalERCModel
from utils.data import collate_conversations, compute_class_weights, load_dataset_bundle
from utils.plotter import plot_confusion_matrix, plot_histories


DATASET_CONFIGS = {
    "iemocap": {
        "epochs": 50,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "hidden_dim": 320,
        "dropout": 0.25,
        "lr": 2e-4,
        "weight_decay": 1e-2,
        "patience": 12,
    },
    "meld": {
        "epochs": 60,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "hidden_dim": 256,
        "dropout": 0.25,
        "lr": 1.5e-4,
        "weight_decay": 5e-3,
        "patience": 12,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled MERC studies.")
    parser.add_argument("--study", choices=["all", "multiseed", "optimizers", "variants"], default="all")
    parser.add_argument("--output_root", type=str, default="checkpoints/controlled_studies")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--datasets", nargs="+", choices=["iemocap", "meld"], default=["iemocap", "meld"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(bundle: dict[str, Any], batch_size: int, num_workers: int = 0, cpu: bool = False) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            bundle[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_conversations,
            pin_memory=torch.cuda.is_available() and not cpu,
        )
        for split in ["train", "val", "test"]
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def flatten_predictions(logits: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    predictions = logits.argmax(dim=-1)
    valid_mask = labels != -100
    return labels[valid_mask].detach().cpu().numpy(), predictions[valid_mask].detach().cpu().numpy()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    text=batch["text"],
                    audio=batch["audio"],
                    visual=batch["visual"],
                    speaker_ids=batch["speaker_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = criterion(outputs["logits"].transpose(1, 2), batch["labels"])
            labels_np, predictions_np = flatten_predictions(outputs["logits"], batch["labels"])
            y_true.append(labels_np)
            y_pred.append(predictions_np)
            losses.append(loss.item())

    y_true_np = np.concatenate(y_true) if y_true else np.asarray([], dtype=np.int64)
    y_pred_np = np.concatenate(y_pred) if y_pred else np.asarray([], dtype=np.int64)
    if y_true_np.size == 0:
        macro_f1 = weighted_f1 = accuracy = 0.0
    else:
        macro_f1 = f1_score(y_true_np, y_pred_np, average="macro")
        weighted_f1 = f1_score(y_true_np, y_pred_np, average="weighted")
        accuracy = accuracy_score(y_true_np, y_pred_np)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy),
        "y_true": y_true_np,
        "y_pred": y_pred_np,
    }


def write_history_csv(history: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history.keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(len(history["epoch"])):
            writer.writerow({key: history[key][row_index] for key in fieldnames})


def optimizer_display_name(name: str) -> str:
    return "MoMask" if name.lower() == "momask" else name.upper()


def flatten_grad_tuple(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for grad, parameter in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros_like(parameter, memory_format=torch.preserve_format).reshape(-1))
        else:
            chunks.append(grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def cosine_from_grads(grad_a: tuple[torch.Tensor | None, ...], grad_b: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> float:
    flat_a = flatten_grad_tuple(grad_a, params)
    flat_b = flatten_grad_tuple(grad_b, params)
    denom = flat_a.norm() * flat_b.norm() + 1e-8
    return float(torch.dot(flat_a, flat_b).item() / denom.item())


def modality_only_losses(
    model: nn.Module,
    batch: dict[str, Any],
    criterion: nn.Module,
    amp_enabled: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_text = torch.zeros_like(batch["text"])
    zero_audio = torch.zeros_like(batch["audio"])
    zero_visual = torch.zeros_like(batch["visual"])
    _, text_loss = forward_loss(model, batch, criterion, batch["text"], zero_audio, zero_visual, amp_enabled, device)
    _, audio_loss = forward_loss(model, batch, criterion, zero_text, batch["audio"], zero_visual, amp_enabled, device)
    _, visual_loss = forward_loss(model, batch, criterion, zero_text, zero_audio, batch["visual"], amp_enabled, device)
    return text_loss, audio_loss, visual_loss


def forward_loss(
    model: nn.Module,
    batch: dict[str, Any],
    criterion: nn.Module,
    text: torch.Tensor,
    audio: torch.Tensor,
    visual: torch.Tensor,
    amp_enabled: bool,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        outputs = model(
            text=text,
            audio=audio,
            visual=visual,
            speaker_ids=batch["speaker_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = criterion(outputs["logits"].transpose(1, 2), batch["labels"])
    return outputs, loss


def build_task_losses(
    model: nn.Module,
    batch: dict[str, Any],
    criterion: nn.Module,
    optimizer_name: str,
    amp_enabled: bool,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, torch.Tensor]]]:
    outputs, main_loss = forward_loss(
        model,
        batch,
        criterion,
        text=batch["text"],
        audio=batch["audio"],
        visual=batch["visual"],
        amp_enabled=amp_enabled,
        device=device,
    )
    task_losses = [("full", main_loss)]
    if optimizer_name in {"pcgrad", "cagrad", "mgda"}:
        zero_text = torch.zeros_like(batch["text"])
        zero_audio = torch.zeros_like(batch["audio"])
        zero_visual = torch.zeros_like(batch["visual"])
        _, text_loss = forward_loss(model, batch, criterion, batch["text"], zero_audio, zero_visual, amp_enabled, device)
        _, audio_loss = forward_loss(model, batch, criterion, zero_text, batch["audio"], zero_visual, amp_enabled, device)
        _, visual_loss = forward_loss(model, batch, criterion, zero_text, zero_audio, batch["visual"], amp_enabled, device)
        task_losses.extend([
            ("text_only", text_loss),
            ("audio_only", audio_loss),
            ("visual_only", visual_loss),
        ])
    return outputs, task_losses


def add_flattened_grad(flat_grad: torch.Tensor, params: list[torch.nn.Parameter]) -> None:
    offset = 0
    for parameter in params:
        numel = parameter.numel()
        grad_chunk = flat_grad[offset : offset + numel].view_as(parameter).to(parameter.dtype)
        if parameter.grad is None:
            parameter.grad = grad_chunk.clone()
        else:
            parameter.grad.add_(grad_chunk)
        offset += numel


def build_model(bundle: dict[str, Any], config: dict[str, Any]) -> MultimodalERCModel:
    return MultimodalERCModel(
        text_dim=bundle["text_dim"],
        audio_dim=bundle["audio_dim"],
        visual_dim=bundle["visual_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=bundle["num_classes"],
        num_speakers=bundle["num_speakers"],
        dropout=config["dropout"],
        use_ev_gate=config["use_ev_gate"],
        ev_gate_type=config["ev_gate_type"],
        ev_gate_distance=config["ev_gate_distance"],
        ev_gate_anchor=config["ev_gate_anchor"],
    )


def build_optimizer(model: nn.Module, config: dict[str, Any]):
    optimizer_name = config["optimizer"]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if optimizer_name == "momask":
        return MoMask(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momask_beta=config["momask_beta"],
            momask_mask_prob=config["momask_mask_prob"],
            momask_momentum_source=config.get("momask_momentum_source", "decoupled"),
        )
    return ConflictOptimizer(
        model.parameters(),
        method=optimizer_name,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )


def train_one_run(
    dataset: str,
    config: dict[str, Any],
    output_dir: Path,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    cpu: bool,
) -> dict[str, Any]:
    set_seed(config["seed"])
    bundle = load_dataset_bundle(dataset, validation_ratio=0.1, seed=config["seed"], protocol=config.get("protocol", "default"), fold_id=config.get("fold_id"))
    dataloaders = create_dataloaders(bundle, batch_size=config["batch_size"], cpu=cpu)
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    amp_enabled = device.type == "cuda" and config["optimizer"] not in {"pcgrad", "cagrad", "mgda"}

    model = build_model(bundle, config).to(device)
    class_weights = compute_class_weights(bundle["train"], bundle["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = build_optimizer(model, config)
    scheduler_target = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(scheduler_target, T_max=max(config["epochs"], 1))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "val_accuracy": [],
        "test_macro_f1": [],
        "test_weighted_f1": [],
        "test_accuracy": [],
        "masked_ratio": [],
        "conflict_ratio": [],
        "learning_rate": [],
        "epoch_time_sec": [],
        "grad_cosine_text_audio": [],
        "grad_cosine_text_visual": [],
        "grad_cosine_audio_visual": [],
    }
    best_state = None
    best_val_weighted_f1 = -math.inf
    epochs_without_improvement = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        epoch_conflicts = []
        progress = tqdm(dataloaders["train"], desc=f"{dataset}-{config['optimizer']}-seed{config['seed']}-epoch{epoch}", leave=False)
        grad_cosines_ta, grad_cosines_tv, grad_cosines_av = [], [], []
        for batch_index, batch in enumerate(progress):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            batch = move_batch(batch, device)
            if config.get("track_grad_cosine", False) and batch_index < int(config.get("conflict_analysis_batches", 4)):
                text_loss, audio_loss, visual_loss = modality_only_losses(model, batch, criterion, False, device)
                text_grads = torch.autograd.grad(text_loss, params, retain_graph=False, allow_unused=True)
                audio_grads = torch.autograd.grad(audio_loss, params, retain_graph=False, allow_unused=True)
                visual_grads = torch.autograd.grad(visual_loss, params, retain_graph=False, allow_unused=True)
                grad_cosines_ta.append(cosine_from_grads(text_grads, audio_grads, params))
                grad_cosines_tv.append(cosine_from_grads(text_grads, visual_grads, params))
                grad_cosines_av.append(cosine_from_grads(audio_grads, visual_grads, params))
                optimizer.zero_grad(set_to_none=True)
            if config["optimizer"] in {"pcgrad", "cagrad", "mgda"}:
                outputs, task_losses = build_task_losses(model, batch, criterion, config["optimizer"], False, device)
                scaled_losses = [loss / config["grad_accum_steps"] for _, loss in task_losses]
                task_grads = [
                    torch.autograd.grad(
                        loss,
                        params,
                        retain_graph=(task_index < len(scaled_losses) - 1),
                        allow_unused=True,
                    )
                    for task_index, loss in enumerate(scaled_losses)
                ]
                combined_grad, conflict_stats = optimizer.combine(task_grads, params)
                add_flattened_grad(combined_grad, params)
                epoch_conflicts.append(conflict_stats.cosine_conflict)
                epoch_losses.append(float(task_losses[0][1].item()))
            else:
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(
                        text=batch["text"],
                        audio=batch["audio"],
                        visual=batch["visual"],
                        speaker_ids=batch["speaker_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = criterion(outputs["logits"].transpose(1, 2), batch["labels"]) / config["grad_accum_steps"]
                scaler.scale(loss).backward()
                epoch_losses.append(float(loss.item() * config["grad_accum_steps"]))

            progress.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")
            should_step = (batch_index + 1) % config["grad_accum_steps"] == 0
            if should_step:
                if config["optimizer"] in {"pcgrad", "cagrad", "mgda"}:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    scaler.unscale_(scheduler_target)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if epoch_losses and len(epoch_losses) % config["grad_accum_steps"] != 0:
            if config["optimizer"] in {"pcgrad", "cagrad", "mgda"}:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                scaler.unscale_(scheduler_target)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        validation_metrics = evaluate(model, dataloaders["val"], criterion, device, amp_enabled, max_eval_batches)
        test_metrics = evaluate(model, dataloaders["test"], criterion, device, amp_enabled, max_eval_batches)
        optimizer_stats = getattr(optimizer, "last_step_stats", {})
        masked_ratio = float(optimizer_stats.get("masked_ratio", 0.0))
        conflict_ratio = float(np.mean(epoch_conflicts)) if epoch_conflicts else float(optimizer_stats.get("conflict_ratio", 0.0))
        learning_rate = float(scheduler_target.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        history["val_loss"].append(validation_metrics["loss"])
        history["val_macro_f1"].append(validation_metrics["macro_f1"])
        history["val_weighted_f1"].append(validation_metrics["weighted_f1"])
        history["val_accuracy"].append(validation_metrics["accuracy"])
        history["test_macro_f1"].append(test_metrics["macro_f1"])
        history["test_weighted_f1"].append(test_metrics["weighted_f1"])
        history["test_accuracy"].append(test_metrics["accuracy"])
        history["masked_ratio"].append(masked_ratio)
        history["conflict_ratio"].append(conflict_ratio)
        history["learning_rate"].append(learning_rate)
        history["epoch_time_sec"].append(float(time.perf_counter() - epoch_start_time))
        history["grad_cosine_text_audio"].append(float(np.mean(grad_cosines_ta)) if grad_cosines_ta else 0.0)
        history["grad_cosine_text_visual"].append(float(np.mean(grad_cosines_tv)) if grad_cosines_tv else 0.0)
        history["grad_cosine_audio_visual"].append(float(np.mean(grad_cosines_av)) if grad_cosines_av else 0.0)

        if validation_metrics["weighted_f1"] > best_val_weighted_f1:
            best_val_weighted_f1 = validation_metrics["weighted_f1"]
            epochs_without_improvement = 0
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "optimizer_name": config["optimizer"],
                "best_val_weighted_f1": best_val_weighted_f1,
                "run_config": config,
            }
        else:
            epochs_without_improvement += 1

        if config["patience"] > 0 and epochs_without_improvement >= config["patience"]:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce any checkpoint state.")

    checkpoint_path = output_dir / "best.pt"
    torch.save(best_state, checkpoint_path)
    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    write_history_csv(history, output_dir / "history.csv")

    model.load_state_dict(best_state["model"])
    final_test_metrics = evaluate(model, dataloaders["test"], criterion, device, amp_enabled, max_eval_batches)
    cm = confusion_matrix(final_test_metrics["y_true"], final_test_metrics["y_pred"], labels=list(range(bundle["num_classes"])))
    optimizer_title = optimizer_display_name(config["optimizer"])
    plot_confusion_matrix(cm, bundle["label_names"], output_dir / "confusion_matrix.pdf", title=f"{dataset.upper()} {optimizer_title}")
    summary = {
        "dataset": dataset,
        "optimizer_name": config["optimizer"],
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(output_dir),
        "test_macro_f1": final_test_metrics["macro_f1"],
        "test_weighted_f1": final_test_metrics["weighted_f1"],
        "test_accuracy": final_test_metrics["accuracy"],
        "best_epoch": int(best_state["epoch"]),
        "best_val_weighted_f1": float(best_state["best_val_weighted_f1"]),
        "run_config": config,
        "mean_epoch_time_sec": float(np.mean(history["epoch_time_sec"])) if history["epoch_time_sec"] else 0.0,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    plot_histories({optimizer_display_name(config["optimizer"]): history}, output_dir / "curves.pdf", metric_key="weighted_f1", metric_label="Weighted-F1")
    return summary


def aggregate_runs(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    weighted = np.array([summary["test_weighted_f1"] for summary in run_summaries], dtype=np.float64)
    macro = np.array([summary["test_macro_f1"] for summary in run_summaries], dtype=np.float64)
    acc = np.array([summary["test_accuracy"] for summary in run_summaries], dtype=np.float64)
    return {
        "count": int(len(run_summaries)),
        "weighted_f1_mean": float(weighted.mean()),
        "weighted_f1_std": float(weighted.std(ddof=1)) if len(weighted) > 1 else 0.0,
        "macro_f1_mean": float(macro.mean()),
        "macro_f1_std": float(macro.std(ddof=1)) if len(macro) > 1 else 0.0,
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "runs": run_summaries,
    }


def maybe_run(dataset: str, config: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if args.resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return train_one_run(dataset, config, output_dir, args.max_train_batches, args.max_eval_batches, args.cpu)


def run_multiseed(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    results = {}
    for dataset in args.datasets:
        dataset_results = {}
        base = DATASET_CONFIGS[dataset]
        for optimizer_name in ["adamw", "momask"]:
            runs = []
            for seed in args.seeds:
                config = {
                    **base,
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "use_ev_gate": True,
                    "ev_gate_type": "scalar",
                    "ev_gate_distance": "l2",
                    "ev_gate_anchor": "text",
                    "momask_beta": 0.9,
                    "momask_mask_prob": 0.35,
                }
                run_dir = output_root / "multiseed" / dataset / f"{optimizer_name}_seed{seed}"
                runs.append(maybe_run(dataset, config, run_dir, args))
            dataset_results[optimizer_name] = aggregate_runs(runs)
        results[dataset] = dataset_results
    path = output_root / "multiseed_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_optimizer_study(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    dataset = "iemocap"
    base = DATASET_CONFIGS[dataset]
    results = {}
    for optimizer_name in ["adamw", "momask", "pcgrad", "cagrad", "mgda"]:
        config = {
            **base,
            "optimizer": optimizer_name,
            "seed": 42,
            "use_ev_gate": True,
            "ev_gate_type": "scalar",
            "ev_gate_distance": "l2",
            "ev_gate_anchor": "text",
            "momask_beta": 0.9,
            "momask_mask_prob": 0.35,
        }
        run_dir = output_root / "optimizer_baselines" / optimizer_name
        results[optimizer_name] = maybe_run(dataset, config, run_dir, args)
    path = output_root / "optimizer_baselines_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_variant_study(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    dataset = "iemocap"
    base = DATASET_CONFIGS[dataset]
    variants = {
        "scalar_text_l2": {"ev_gate_type": "scalar", "ev_gate_distance": "l2", "ev_gate_anchor": "text"},
        "vector_text_l2": {"ev_gate_type": "vector", "ev_gate_distance": "l2", "ev_gate_anchor": "text"},
        "vector_text_cosine": {"ev_gate_type": "vector", "ev_gate_distance": "cosine", "ev_gate_anchor": "text"},
        "vector_learned_cosine": {"ev_gate_type": "vector", "ev_gate_distance": "cosine", "ev_gate_anchor": "learned"},
    }
    results = {}
    for variant_name, variant in variants.items():
        config = {
            **base,
            "optimizer": "momask",
            "seed": 42,
            "use_ev_gate": True,
            **variant,
            "momask_beta": 0.9,
            "momask_mask_prob": 0.35,
        }
        run_dir = output_root / "ev_gate_variants" / variant_name
        results[variant_name] = maybe_run(dataset, config, run_dir, args)
    path = output_root / "ev_gate_variants_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if args.study in {"all", "multiseed"}:
        all_results["multiseed"] = run_multiseed(args, output_root)
    if args.study in {"all", "optimizers"}:
        all_results["optimizers"] = run_optimizer_study(args, output_root)
    if args.study in {"all", "variants"}:
        all_results["variants"] = run_variant_study(args, output_root)
    (output_root / "all_controlled_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
