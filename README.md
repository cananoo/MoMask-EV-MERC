# MoMask-EV-MERC

PyTorch code for "Stable Multimodal Emotion Recognition in Conversation via Expectancy-Violation Routing and Momentum-Aligned Masking".

## Overview

This repository contains the training and evaluation code for a lightweight MERC pipeline built around:

- `EV-Gate` for text-anchored multimodal fusion
- `MoMask` for momentum-aligned masking on top of `AdamW`

## Setup

Recommended Python version: `3.10+`

```bash
pip install -r requirements.txt
```

## Data

This repository does not include processed features or model checkpoints.

Place the local feature bundles at:

- `dataset/iemocap_multimodal_features.pkl`
- `dataset/meld_multimodal_features.pkl`

Use the datasets in accordance with their original licenses.

## Training

IEMOCAP:

```bash
python train.py --dataset iemocap --use_ev_gate --use_momask
```

MELD:

```bash
python train.py --dataset meld --use_ev_gate --use_momask
```

## Reproducibility

- `tools/run_controlled_studies.py` runs multi-seed and optimizer comparison experiments.
- `tools/robustness_eval.py` evaluates robustness under modality corruption.

## Repository structure

- `train.py` main training entry point
- `models/` model components and optimizer wrappers
- `utils/` data loading and plotting utilities
- `tools/` experiment scripts used for reproduction

## Notes

- Large artifacts such as checkpoints, cached features, figures, and manuscript files are excluded from version control.
