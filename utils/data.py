from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


IEMOCAP_LABELS = ["hap", "sad", "neu", "ang", "exc", "fru"]
MELD_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


@dataclass
class ConversationSample:
    conversation_id: str
    text: np.ndarray
    audio: np.ndarray
    visual: np.ndarray
    labels: np.ndarray
    speaker_ids: np.ndarray
    utterance_ids: list[str]
    utterances: list[str]


class ConversationDataset(Dataset):
    def __init__(self, samples: list[ConversationSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ConversationSample:
        return self.samples[index]


def _stack_text_streams(text_streams: list[list[np.ndarray]]) -> np.ndarray:
    stacked = np.stack([np.stack(stream, axis=0) for stream in text_streams], axis=0)
    return stacked.mean(axis=0).astype(np.float32)


def _iemocap_speaker_ids(speakers: list[str]) -> np.ndarray:
    mapping = {"M": 0, "F": 1}
    return np.asarray([mapping.get(speaker, 0) for speaker in speakers], dtype=np.int64)


def _meld_speaker_ids(raw_speakers: list[list[int]] | list[np.ndarray]) -> np.ndarray:
    speaker_ids = []
    for speaker in raw_speakers:
        speaker_array = np.asarray(speaker)
        if speaker_array.ndim == 0:
            speaker_ids.append(int(speaker_array))
        else:
            speaker_ids.append(int(np.argmax(speaker_array)))
    return np.asarray(speaker_ids, dtype=np.int64)


def _split_train_val(
    conversation_ids: list[Any],
    validation_ratio: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    ordered_ids = list(conversation_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered_ids)
    validation_size = max(1, int(len(ordered_ids) * validation_ratio))
    validation_ids = ordered_ids[:validation_size]
    train_ids = ordered_ids[validation_size:]
    return train_ids, validation_ids


def load_dataset_bundle(
    dataset_name: str,
    dataset_dir: str | Path = "dataset",
    validation_ratio: float = 0.1,
    seed: int = 42,
    protocol: str = "default",
    fold_id: int | None = None,
) -> dict[str, Any]:
    dataset_name = dataset_name.lower()
    dataset_dir = Path(dataset_dir)
    if dataset_name == "iemocap":
        path = dataset_dir / "iemocap_multimodal_features.pkl"
        raw_data = _load_pickle(path)
        return _build_iemocap_bundle(raw_data, validation_ratio, seed, protocol=protocol, fold_id=fold_id)
    if dataset_name == "meld":
        path = dataset_dir / "meld_multimodal_features.pkl"
        raw_data = _load_pickle(path)
        return _build_meld_bundle(raw_data, validation_ratio, seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _build_iemocap_bundle(
    raw_data: list[Any],
    validation_ratio: float,
    seed: int,
    protocol: str = "default",
    fold_id: int | None = None,
) -> dict[str, Any]:
    if protocol == "default":
        train_ids, validation_ids = _split_train_val(raw_data[10], validation_ratio, seed)
        test_ids = list(raw_data[11])
    elif protocol == "session_5fold":
        if fold_id is None or fold_id not in {1, 2, 3, 4, 5}:
            raise ValueError("session_5fold protocol requires fold_id in {1,2,3,4,5}.")
        all_ids = list(raw_data[10]) + list(raw_data[11])
        fold_prefix = f"Ses0{fold_id}"
        test_ids = [conversation_id for conversation_id in all_ids if str(conversation_id).startswith(fold_prefix)]
        remaining_ids = [conversation_id for conversation_id in all_ids if not str(conversation_id).startswith(fold_prefix)]
        train_ids, validation_ids = _split_train_val(remaining_ids, validation_ratio, seed)
    else:
        raise ValueError(f"Unsupported IEMOCAP protocol: {protocol}")
    samples = {}
    for conversation_id in set(train_ids + validation_ids + test_ids):
        text = _stack_text_streams([
            raw_data[3][conversation_id],
            raw_data[4][conversation_id],
            raw_data[5][conversation_id],
            raw_data[6][conversation_id],
        ])
        audio = np.asarray(raw_data[7][conversation_id], dtype=np.float32)
        visual = np.stack(raw_data[8][conversation_id], axis=0).astype(np.float32)
        labels = np.asarray(raw_data[2][conversation_id], dtype=np.int64)
        speaker_ids = _iemocap_speaker_ids(raw_data[1][conversation_id])
        samples[conversation_id] = ConversationSample(
            conversation_id=conversation_id,
            text=text,
            audio=audio,
            visual=visual,
            labels=labels,
            speaker_ids=speaker_ids,
            utterance_ids=list(raw_data[0][conversation_id]),
            utterances=list(raw_data[9][conversation_id]),
        )
    return {
        "train": ConversationDataset([samples[conversation_id] for conversation_id in train_ids]),
        "val": ConversationDataset([samples[conversation_id] for conversation_id in validation_ids]),
        "test": ConversationDataset([samples[conversation_id] for conversation_id in test_ids]),
        "label_names": IEMOCAP_LABELS,
        "text_dim": 1024,
        "audio_dim": 1582,
        "visual_dim": 342,
        "num_speakers": 2,
        "num_classes": len(IEMOCAP_LABELS),
        "protocol": protocol,
        "fold_id": fold_id,
    }


def _build_meld_bundle(raw_data: list[Any], validation_ratio: float, seed: int) -> dict[str, Any]:
    train_ids, validation_ids = _split_train_val(sorted(list(raw_data[10])), validation_ratio, seed)
    test_ids = sorted(list(raw_data[11]))
    samples = {}
    for conversation_id in set(train_ids + validation_ids + test_ids):
        text = _stack_text_streams([
            raw_data[3][conversation_id],
            raw_data[4][conversation_id],
            raw_data[5][conversation_id],
            raw_data[6][conversation_id],
        ])
        audio = np.asarray(raw_data[7][conversation_id], dtype=np.float32)
        visual = np.asarray(raw_data[8][conversation_id], dtype=np.float32)
        labels = np.asarray(raw_data[2][conversation_id], dtype=np.int64)
        speaker_ids = _meld_speaker_ids(raw_data[1][conversation_id])
        utterance_ids = [f"{conversation_id}_{turn_index}" for turn_index in range(len(labels))]
        samples[conversation_id] = ConversationSample(
            conversation_id=str(conversation_id),
            text=text,
            audio=audio,
            visual=visual,
            labels=labels,
            speaker_ids=speaker_ids,
            utterance_ids=utterance_ids,
            utterances=list(raw_data[9][conversation_id]),
        )
    return {
        "train": ConversationDataset([samples[conversation_id] for conversation_id in train_ids]),
        "val": ConversationDataset([samples[conversation_id] for conversation_id in validation_ids]),
        "test": ConversationDataset([samples[conversation_id] for conversation_id in test_ids]),
        "label_names": MELD_LABELS,
        "text_dim": 1024,
        "audio_dim": 300,
        "visual_dim": 342,
        "num_speakers": 9,
        "num_classes": len(MELD_LABELS),
        "protocol": "default",
        "fold_id": None,
    }


def collate_conversations(batch: list[ConversationSample]) -> dict[str, torch.Tensor | list[str] | list[list[str]]]:
    batch_size = len(batch)
    max_seq_len = max(sample.labels.shape[0] for sample in batch)
    text_dim = batch[0].text.shape[-1]
    audio_dim = batch[0].audio.shape[-1]
    visual_dim = batch[0].visual.shape[-1]

    text = torch.zeros(batch_size, max_seq_len, text_dim, dtype=torch.float32)
    audio = torch.zeros(batch_size, max_seq_len, audio_dim, dtype=torch.float32)
    visual = torch.zeros(batch_size, max_seq_len, visual_dim, dtype=torch.float32)
    labels = torch.full((batch_size, max_seq_len), fill_value=-100, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    conversation_ids: list[str] = []
    utterance_ids: list[list[str]] = []
    utterances: list[list[str]] = []
    for batch_index, sample in enumerate(batch):
        seq_len = sample.labels.shape[0]
        text[batch_index, :seq_len] = torch.from_numpy(sample.text)
        audio[batch_index, :seq_len] = torch.from_numpy(sample.audio)
        visual[batch_index, :seq_len] = torch.from_numpy(sample.visual)
        labels[batch_index, :seq_len] = torch.from_numpy(sample.labels)
        speaker_ids[batch_index, :seq_len] = torch.from_numpy(sample.speaker_ids)
        attention_mask[batch_index, :seq_len] = True
        lengths[batch_index] = seq_len
        conversation_ids.append(sample.conversation_id)
        utterance_ids.append(sample.utterance_ids)
        utterances.append(sample.utterances)

    return {
        "text": text,
        "audio": audio,
        "visual": visual,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "conversation_ids": conversation_ids,
        "utterance_ids": utterance_ids,
        "utterances": utterances,
    }


def compute_class_weights(dataset: ConversationDataset, num_classes: int) -> torch.Tensor:
    label_counts = np.zeros(num_classes, dtype=np.float64)
    for sample in dataset.samples:
        for label in sample.labels:
            label_counts[int(label)] += 1.0
    label_counts = np.maximum(label_counts, 1.0)
    weights = label_counts.sum() / (num_classes * label_counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
