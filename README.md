# MoMask-EV-MERC

Official PyTorch implementation for “Stable Multimodal Emotion Recognition in Conversation via Expectancy-Violation Routing and Momentum-Aligned Masking”.

This repository contains the training code, model definitions, and analysis scripts for the MERC experiments.

## Overview

Lightweight multimodal emotion recognition in conversation with:

- `EV-Gate`: expectancy-violation-based fusion
- `MoMask`: momentum-aligned masking on top of `AdamW`

## Repository layout

- `train.py` — main training entry point
- `models/` — MERC model, EV-Gate, MoMask optimizer wrapper, LoRA utilities
- `utils/` — data loading and plotting helpers
- `tools/` — reviewer-study, robustness, and manuscript-analysis scripts
- `figures/` — generated experiment figures

## Data

The repository does not ship the processed feature bundles or model checkpoints.

Expected local data files:

- `dataset/iemocap_multimodal_features.pkl`
- `dataset/meld_multimodal_features.pkl`

These are ignored by Git and should be prepared locally according to the dataset licenses.

## Environment

Recommended Python version: `3.10+`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Example IEMOCAP run:

```bash
python train.py --dataset iemocap --use_ev_gate --use_momask
```

Example MELD run:

```bash
python train.py --dataset meld --use_ev_gate --use_momask
```

## Notes

- Large checkpoints, processed feature bundles, manuscript sources, and local debugging images are intentionally excluded.
- If you plan to reproduce the paper numbers, prepare the local feature bundles under `dataset/` and write outputs under `checkpoints/`.
