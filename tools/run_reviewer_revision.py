from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_controlled_studies import DATASET_CONFIGS, aggregate_runs, maybe_run


class ArgsProxy:
    def __init__(self, resume: bool = True, max_train_batches: int | None = None, max_eval_batches: int | None = None, cpu: bool = False):
        self.resume = resume
        self.max_train_batches = max_train_batches
        self.max_eval_batches = max_eval_batches
        self.cpu = cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run high-priority reviewer revision experiments.")
    parser.add_argument("--study", choices=["all", "iemocap_cv", "meld_optimizers", "sensitivity", "grad_conflict"], default="all")
    parser.add_argument("--output_root", type=str, default="checkpoints/reviewer_revision")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def make_args_proxy(args: argparse.Namespace) -> ArgsProxy:
    return ArgsProxy(resume=args.resume, max_train_batches=args.max_train_batches, max_eval_batches=args.max_eval_batches, cpu=args.cpu)


def run_iemocap_session_cv(args: argparse.Namespace, output_root: Path) -> dict:
    proxy = make_args_proxy(args)
    base = DATASET_CONFIGS['iemocap']
    results = {}
    for optimizer_name in ['adamw', 'momask']:
        runs = []
        for fold_id in [1, 2, 3, 4, 5]:
            config = {
                **base,
                'optimizer': optimizer_name,
                'seed': 42,
                'use_ev_gate': True,
                'ev_gate_type': 'scalar',
                'ev_gate_distance': 'l2',
                'ev_gate_anchor': 'text',
                'momask_beta': 0.9,
                'momask_mask_prob': 0.35,
                'protocol': 'session_5fold',
                'fold_id': fold_id,
            }
            run_dir = output_root / 'iemocap_session5fold' / optimizer_name / f'fold{fold_id}'
            runs.append(maybe_run('iemocap', config, run_dir, proxy))
        results[optimizer_name] = aggregate_runs(runs)
    out = output_root / 'iemocap_session5fold_summary.json'
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def run_meld_optimizer_baselines(args: argparse.Namespace, output_root: Path) -> dict:
    proxy = make_args_proxy(args)
    base = DATASET_CONFIGS['meld']
    results = {}
    for optimizer_name in ['adamw', 'momask', 'pcgrad', 'cagrad', 'mgda']:
        config = {
            **base,
            'optimizer': optimizer_name,
            'seed': 42,
            'use_ev_gate': True,
            'ev_gate_type': 'scalar',
            'ev_gate_distance': 'l2',
            'ev_gate_anchor': 'text',
            'momask_beta': 0.9,
            'momask_mask_prob': 0.35,
            'protocol': 'default',
            'fold_id': None,
        }
        run_dir = output_root / 'meld_optimizer_baselines' / optimizer_name
        results[optimizer_name] = maybe_run('meld', config, run_dir, proxy)
    out = output_root / 'meld_optimizer_baselines_summary.json'
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def run_sensitivity(args: argparse.Namespace, output_root: Path) -> dict:
    proxy = make_args_proxy(args)
    configs = {
        'iemocap': {
            **DATASET_CONFIGS['iemocap'],
            'protocol': 'session_5fold',
            'fold_id': 5,
        },
        'meld': {
            **DATASET_CONFIGS['meld'],
            'protocol': 'default',
            'fold_id': None,
        },
    }
    results = {}
    for dataset, base in configs.items():
        dataset_results = {'mask_prob': {}, 'beta': {}}
        for p in [0.1, 0.3, 0.5, 0.7]:
            config = {
                **base,
                'optimizer': 'momask',
                'seed': 42,
                'use_ev_gate': True,
                'ev_gate_type': 'scalar',
                'ev_gate_distance': 'l2',
                'ev_gate_anchor': 'text',
                'momask_beta': 0.9,
                'momask_mask_prob': p,
            }
            run_dir = output_root / 'sensitivity' / dataset / f'maskprob_{str(p).replace(".", "p")}'
            dataset_results['mask_prob'][str(p)] = maybe_run(dataset, config, run_dir, proxy)
        for beta in [0.8, 0.9, 0.99]:
            config = {
                **base,
                'optimizer': 'momask',
                'seed': 42,
                'use_ev_gate': True,
                'ev_gate_type': 'scalar',
                'ev_gate_distance': 'l2',
                'ev_gate_anchor': 'text',
                'momask_beta': beta,
                'momask_mask_prob': 0.35,
            }
            run_dir = output_root / 'sensitivity' / dataset / f'beta_{str(beta).replace(".", "p")}'
            dataset_results['beta'][str(beta)] = maybe_run(dataset, config, run_dir, proxy)
        results[dataset] = dataset_results
    out = output_root / 'sensitivity_summary.json'
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def run_grad_conflict(args: argparse.Namespace, output_root: Path) -> dict:
    proxy = make_args_proxy(args)
    base = {
        **DATASET_CONFIGS['iemocap'],
        'protocol': 'session_5fold',
        'fold_id': 5,
        'track_grad_cosine': True,
        'conflict_analysis_batches': 8,
    }
    results = {}
    for optimizer_name in ['adamw', 'momask']:
        config = {
            **base,
            'optimizer': optimizer_name,
            'seed': 42,
            'use_ev_gate': True,
            'ev_gate_type': 'scalar',
            'ev_gate_distance': 'l2',
            'ev_gate_anchor': 'text',
            'momask_beta': 0.9,
            'momask_mask_prob': 0.35,
        }
        run_dir = output_root / 'gradient_conflict' / optimizer_name
        results[optimizer_name] = maybe_run('iemocap', config, run_dir, proxy)
    out = output_root / 'gradient_conflict_summary.json'
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def build_overhead_summary(output_root: Path) -> dict:
    summary = {}
    for path in output_root.rglob('summary.json'):
        if path.name != 'summary.json':
            continue
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if isinstance(obj, dict) and 'optimizer_name' in obj:
            summary[str(path)] = {
                'dataset': obj.get('dataset'),
                'optimizer_name': obj.get('optimizer_name'),
                'mean_epoch_time_sec': obj.get('mean_epoch_time_sec', 0.0),
                'test_weighted_f1': obj.get('test_weighted_f1', 0.0),
            }
    out = output_root / 'epoch_time_overview.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.study in {'all', 'iemocap_cv'}:
        results['iemocap_session5fold'] = run_iemocap_session_cv(args, output_root)
    if args.study in {'all', 'meld_optimizers'}:
        results['meld_optimizer_baselines'] = run_meld_optimizer_baselines(args, output_root)
    if args.study in {'all', 'sensitivity'}:
        results['sensitivity'] = run_sensitivity(args, output_root)
    if args.study in {'all', 'grad_conflict'}:
        results['gradient_conflict'] = run_grad_conflict(args, output_root)
    results['epoch_overhead'] = build_overhead_summary(output_root)
    (output_root / 'all_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
