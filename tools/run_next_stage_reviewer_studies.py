from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next-stage reviewer-driven studies.")
    parser.add_argument(
        "--study",
        choices=["all", "lora_compare", "ema_ablation", "gate_position"],
        default="all",
    )
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default="iemocap")
    parser.add_argument("--output_root", type=str, default="checkpoints/next_stage_revision")
    parser.add_argument("--model_name", type=str, default="roberta-large")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    pretty = " ".join(command)
    print(pretty)
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=ROOT)


def base_command(args: argparse.Namespace, run_name: str) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "--dataset",
        args.dataset,
        "--run_name",
        run_name,
        "--output_dir",
        args.output_root,
        "--text_encoder_mode",
        "lora",
        "--text_encoder_name",
        args.model_name,
        "--text_lora_rank",
        "8",
        "--text_lora_alpha",
        "16",
        "--text_lora_dropout",
        "0.05",
        "--text_max_length",
        "64",
        "--gate_position",
        "post_context",
    ]
    if args.max_train_batches is not None:
        command.extend(["--max_train_batches", str(args.max_train_batches)])
    if args.max_eval_batches is not None:
        command.extend(["--max_eval_batches", str(args.max_eval_batches)])
    if args.cpu:
        command.append("--cpu")
    return command


def run_lora_compare(args: argparse.Namespace) -> None:
    for use_momask in [False, True]:
        run_name = f"{args.dataset}_lora_postctx_{'momask' if use_momask else 'adamw'}"
        command = base_command(args, run_name)
        command.extend(["--use_momask" if use_momask else "--no-use_momask"])
        run_command(command, args.dry_run)


def run_ema_ablation(args: argparse.Namespace) -> None:
    for momentum_source in ["adamw_expavg", "decoupled"]:
        run_name = f"{args.dataset}_lora_ema_{momentum_source}"
        command = base_command(args, run_name)
        command.extend(
            [
                "--use_momask",
                "--momask_momentum_source",
                momentum_source,
            ]
        )
        run_command(command, args.dry_run)


def run_gate_position(args: argparse.Namespace) -> None:
    for gate_position in ["pre_context", "post_context"]:
        run_name = f"{args.dataset}_lora_{gate_position}_momask"
        command = base_command(args, run_name)
        command[command.index("--gate_position") + 1] = gate_position
        command.extend(["--use_momask"])
        run_command(command, args.dry_run)


def main() -> None:
    args = parse_args()
    if args.study in {"all", "lora_compare"}:
        run_lora_compare(args)
    if args.study in {"all", "ema_ablation"}:
        run_ema_ablation(args)
    if args.study in {"all", "gate_position"}:
        run_gate_position(args)


if __name__ == "__main__":
    main()
