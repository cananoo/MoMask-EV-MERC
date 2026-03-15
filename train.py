from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.momask_optim import MoMask
from models.merc_model import MultimodalERCModel
from utils.data import collate_conversations, compute_class_weights, load_dataset_bundle
from utils.plotter import (
    plot_confusion_matrix,
    plot_histories,
    plot_layerwise_masking,
    write_architecture_figure,
    write_architecture_mermaid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MERC with EV-Gate and MoMask.")
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default="iemocap")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--momask_beta", type=float, default=0.9)
    parser.add_argument("--momask_mask_prob", type=float, default=0.35)
    parser.add_argument("--momask_momentum_source", choices=["decoupled", "adamw_expavg"], default="decoupled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--compare_optimizers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_ev_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ev_gate_type", choices=["scalar", "vector"], default="scalar")
    parser.add_argument("--ev_gate_distance", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--ev_gate_anchor", choices=["text", "mean", "learned"], default="text")
    parser.add_argument("--gate_position", choices=["pre_context", "post_context"], default="pre_context")
    parser.add_argument("--text_encoder_mode", choices=["offline", "lora"], default="offline")
    parser.add_argument("--text_encoder_name", type=str, default="roberta-large")
    parser.add_argument("--text_lora_rank", type=int, default=8)
    parser.add_argument("--text_lora_alpha", type=int, default=16)
    parser.add_argument("--text_lora_dropout", type=float, default=0.05)
    parser.add_argument("--text_max_length", type=int, default=64)
    parser.add_argument("--text_pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--text_train_layer_norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_momask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(bundle: dict[str, Any], args: argparse.Namespace) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            bundle[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            collate_fn=collate_conversations,
            pin_memory=torch.cuda.is_available() and not args.cpu,
        )
        for split in ["train", "val", "test"]
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def flatten_predictions(logits: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    predictions = logits.argmax(dim=-1)
    valid_mask = labels != -100
    return (
        labels[valid_mask].detach().cpu().numpy(),
        predictions[valid_mask].detach().cpu().numpy(),
    )


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
                    utterances=batch.get("utterances"),
                )
                loss = criterion(outputs["logits"].transpose(1, 2), batch["labels"])

            labels_np, predictions_np = flatten_predictions(outputs["logits"], batch["labels"])
            y_true.append(labels_np)
            y_pred.append(predictions_np)
            losses.append(loss.item())

    y_true_np = np.concatenate(y_true) if y_true else np.asarray([], dtype=np.int64)
    y_pred_np = np.concatenate(y_pred) if y_pred else np.asarray([], dtype=np.int64)
    macro_f1 = f1_score(y_true_np, y_pred_np, average="macro") if y_true_np.size else 0.0
    weighted_f1 = f1_score(y_true_np, y_pred_np, average="weighted") if y_true_np.size else 0.0
    accuracy = accuracy_score(y_true_np, y_pred_np) if y_true_np.size else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy),
        "y_true": y_true_np,
        "y_pred": y_pred_np,
    }


def write_history_csv(history: dict[str, list[float]], csv_path: Path) -> None:
    fieldnames = list(history.keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in zip(*[history[key] for key in fieldnames]):
            writer.writerow({fieldname: value for fieldname, value in zip(fieldnames, row)})


def optimizer_display_name(name: str) -> str:
    return "MoMask" if name.lower() == "momask" else name.upper()


def summarize_parameters(model: nn.Module) -> dict[str, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    lora_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "lora_trainable_params": int(lora_params),
    }


def train_one_run(
    tag: str,
    bundle: dict[str, Any],
    dataloaders: dict[str, DataLoader],
    args: argparse.Namespace,
    output_root: Path,
    force_use_momask: bool | None = None,
) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    amp_enabled = device.type == "cuda"
    model = MultimodalERCModel(
        text_dim=bundle["text_dim"],
        audio_dim=bundle["audio_dim"],
        visual_dim=bundle["visual_dim"],
        hidden_dim=args.hidden_dim,
        num_classes=bundle["num_classes"],
        num_speakers=bundle["num_speakers"],
        dropout=args.dropout,
        use_ev_gate=args.use_ev_gate,
        ev_gate_type=args.ev_gate_type,
        ev_gate_distance=args.ev_gate_distance,
        ev_gate_anchor=args.ev_gate_anchor,
        text_encoder_mode=args.text_encoder_mode,
        text_encoder_name=args.text_encoder_name,
        text_lora_rank=args.text_lora_rank,
        text_lora_alpha=args.text_lora_alpha,
        text_lora_dropout=args.text_lora_dropout,
        text_max_length=args.text_max_length,
        text_pooling=args.text_pooling,
        text_train_layer_norm=args.text_train_layer_norm,
        gate_position=args.gate_position,
    ).to(device)
    parameter_summary = summarize_parameters(model)

    class_weights = compute_class_weights(bundle["train"], bundle["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    use_momask = args.use_momask if force_use_momask is None else force_use_momask
    if use_momask:
        optimizer = MoMask(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momask_beta=args.momask_beta,
            momask_mask_prob=args.momask_mask_prob,
            momask_momentum_source=args.momask_momentum_source,
        )
        optimizer.register_parameter_names(model.named_parameters())
        optimizer_name = "momask"
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_name = "adamw"

    scheduler_target = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(scheduler_target, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
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
    }
    best_state = None
    best_val_weighted_f1 = -math.inf
    epochs_without_improvement = 0
    layerwise_history: list[dict[str, dict[str, float]]] = []
    run_dir = output_root / f"{tag}_{optimizer_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        epoch_layerwise_totals: dict[str, dict[str, float]] = {}
        progress = tqdm(dataloaders["train"], desc=f"{tag}-{optimizer_name}-epoch{epoch}", leave=False)
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            batch = move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    text=batch["text"],
                    audio=batch["audio"],
                    visual=batch["visual"],
                    speaker_ids=batch["speaker_ids"],
                    attention_mask=batch["attention_mask"],
                    utterances=batch.get("utterances"),
                )
                loss = criterion(outputs["logits"].transpose(1, 2), batch["labels"])
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_index + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(scheduler_target)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_stats = getattr(optimizer, "last_step_stats", {})
                for bucket, bucket_stats in optimizer_stats.get("layerwise", {}).items():
                    aggregate = epoch_layerwise_totals.setdefault(
                        bucket,
                        {
                            "masked_elements": 0.0,
                            "conflict_elements": 0.0,
                            "total_elements": 0.0,
                        },
                    )
                    aggregate["masked_elements"] += float(bucket_stats.get("masked_elements", 0.0))
                    aggregate["conflict_elements"] += float(bucket_stats.get("conflict_elements", 0.0))
                    aggregate["total_elements"] += float(bucket_stats.get("total_elements", 0.0))

            epoch_losses.append(loss.item() * args.grad_accum_steps)
            progress.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")

        if epoch_losses and len(epoch_losses) % args.grad_accum_steps != 0:
            scaler.unscale_(scheduler_target)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_stats = getattr(optimizer, "last_step_stats", {})
            for bucket, bucket_stats in optimizer_stats.get("layerwise", {}).items():
                aggregate = epoch_layerwise_totals.setdefault(
                    bucket,
                    {
                        "masked_elements": 0.0,
                        "conflict_elements": 0.0,
                        "total_elements": 0.0,
                    },
                )
                aggregate["masked_elements"] += float(bucket_stats.get("masked_elements", 0.0))
                aggregate["conflict_elements"] += float(bucket_stats.get("conflict_elements", 0.0))
                aggregate["total_elements"] += float(bucket_stats.get("total_elements", 0.0))

        scheduler.step()

        validation_metrics = evaluate(model, dataloaders["val"], criterion, device, amp_enabled, args.max_eval_batches)
        test_metrics = evaluate(model, dataloaders["test"], criterion, device, amp_enabled, args.max_eval_batches)
        optimizer_stats = getattr(optimizer, "last_step_stats", {})
        masked_ratio = float(optimizer_stats.get("masked_ratio", 0.0))
        conflict_ratio = float(optimizer_stats.get("conflict_ratio", 0.0))
        learning_rate = float(scheduler_target.param_groups[0]["lr"])
        epoch_time_sec = float(time.perf_counter() - epoch_start_time)
        epoch_layerwise_ratios = {
            bucket: {
                "masked_ratio": stats["masked_elements"] / max(stats["total_elements"], 1.0),
                "conflict_ratio": stats["conflict_elements"] / max(stats["total_elements"], 1.0),
            }
            for bucket, stats in epoch_layerwise_totals.items()
            if stats["total_elements"] > 0
        }
        layerwise_history.append(epoch_layerwise_ratios)

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
        history["epoch_time_sec"].append(epoch_time_sec)

        if validation_metrics["weighted_f1"] > best_val_weighted_f1:
            best_val_weighted_f1 = validation_metrics["weighted_f1"]
            epochs_without_improvement = 0
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "optimizer_name": optimizer_name,
                "best_val_weighted_f1": best_val_weighted_f1,
                "run_config": {
                    "dataset": args.dataset,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "use_ev_gate": args.use_ev_gate,
                    "ev_gate_type": args.ev_gate_type,
                    "ev_gate_distance": args.ev_gate_distance,
                    "ev_gate_anchor": args.ev_gate_anchor,
                    "gate_position": args.gate_position,
                    "text_encoder_mode": args.text_encoder_mode,
                    "text_encoder_name": args.text_encoder_name,
                    "text_lora_rank": args.text_lora_rank,
                    "text_lora_alpha": args.text_lora_alpha,
                    "text_lora_dropout": args.text_lora_dropout,
                    "text_max_length": args.text_max_length,
                    "text_pooling": args.text_pooling,
                    "text_train_layer_norm": args.text_train_layer_norm,
                    "total_params": parameter_summary["total_params"],
                    "trainable_params": parameter_summary["trainable_params"],
                    "lora_trainable_params": parameter_summary["lora_trainable_params"],
                    "momask_beta": args.momask_beta,
                    "momask_mask_prob": args.momask_mask_prob,
                    "momask_momentum_source": args.momask_momentum_source,
                    "seed": args.seed,
                },
                "layerwise_history": layerwise_history,
            }
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce any checkpoint state.")

    checkpoint_path = run_dir / "best.pt"
    torch.save(best_state, checkpoint_path)
    with (run_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (run_dir / "layerwise_masking.json").open("w", encoding="utf-8") as file:
        json.dump(layerwise_history, file, indent=2)
    write_history_csv(history, run_dir / "history.csv")
    plot_layerwise_masking(layerwise_history, run_dir / "layerwise_masking.pdf")

    model.load_state_dict(best_state["model"])
    final_test_metrics = evaluate(model, dataloaders["test"], criterion, device, amp_enabled, args.max_eval_batches)
    cm = confusion_matrix(final_test_metrics["y_true"], final_test_metrics["y_pred"], labels=list(range(bundle["num_classes"])))
    optimizer_title = optimizer_display_name(optimizer_name)
    plot_confusion_matrix(cm, bundle["label_names"], run_dir / "confusion_matrix.pdf", title=f"{args.dataset.upper()} {optimizer_title}")
    return {
        "run_dir": str(run_dir),
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "optimizer_name": optimizer_name,
        "confusion_matrix": cm.tolist(),
        "test_macro_f1": final_test_metrics["macro_f1"],
        "test_weighted_f1": final_test_metrics["weighted_f1"],
        "test_accuracy": final_test_metrics["accuracy"],
        "best_epoch": int(best_state["epoch"]),
        "best_val_weighted_f1": float(best_state["best_val_weighted_f1"]),
        "run_config": best_state.get("run_config", {}),
        "layerwise_history": layerwise_history,
        "parameter_summary": parameter_summary,
        "mean_epoch_time_sec": float(np.mean(history["epoch_time_sec"])) if history["epoch_time_sec"] else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    bundle = load_dataset_bundle(args.dataset, validation_ratio=args.validation_ratio, seed=args.seed)
    dataloaders = create_dataloaders(bundle, args)
    run_name = args.run_name or f"{args.dataset}_ev{int(args.use_ev_gate)}_mm{int(args.use_momask)}"
    output_root = Path(args.output_dir) / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    write_architecture_mermaid(Path("figures") / "evgate_momask_architecture.mmd")
    write_architecture_figure(Path("figures") / "evgate_momask_architecture.pdf")

    if args.compare_optimizers:
        adamw_result = train_one_run("baseline", bundle, dataloaders, args, output_root, force_use_momask=False)
        momask_result = train_one_run("proposed", bundle, dataloaders, args, output_root, force_use_momask=True)
        plot_histories(
            {
                "AdamW": adamw_result["history"],
                "MoMask": momask_result["history"],
            },
            Path("figures") / f"{args.dataset}_optimizer_comparison.pdf",
            metric_key="weighted_f1",
            metric_label="Weighted-F1",
        )
        summary = {
            "adamw": {
                "test_macro_f1": adamw_result["test_macro_f1"],
                "test_weighted_f1": adamw_result["test_weighted_f1"],
                "test_accuracy": adamw_result["test_accuracy"],
                "best_epoch": adamw_result["best_epoch"],
                "run_dir": adamw_result["run_dir"],
                "run_config": adamw_result["run_config"],
            },
            "momask": {
                "test_macro_f1": momask_result["test_macro_f1"],
                "test_weighted_f1": momask_result["test_weighted_f1"],
                "test_accuracy": momask_result["test_accuracy"],
                "best_epoch": momask_result["best_epoch"],
                "run_dir": momask_result["run_dir"],
                "run_config": momask_result["run_config"],
            },
        }
    else:
        result = train_one_run("single", bundle, dataloaders, args, output_root)
        optimizer_label = optimizer_display_name(result["optimizer_name"])
        plot_histories(
            {optimizer_label: result["history"]},
            Path("figures") / f"{args.dataset}_{result['optimizer_name']}_curves.pdf",
            metric_key="weighted_f1",
            metric_label="Weighted-F1",
        )
        summary = {
            result["optimizer_name"]: {
                "test_macro_f1": result["test_macro_f1"],
                "test_weighted_f1": result["test_weighted_f1"],
                "test_accuracy": result["test_accuracy"],
                "best_epoch": result["best_epoch"],
                "run_dir": result["run_dir"],
                "run_config": result["run_config"],
            }
        }

    with (output_root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
