"""Partial fine-tuning of the FaceNet backbone (decisions D2 and D5)."""

from facial_recognition_system.training.arcface import ArcFaceHead
from facial_recognition_system.training.data import (
    FaceDataset,
    build_augmentation,
    label_indices,
    select_training_subset,
)
from facial_recognition_system.training.freezing import freeze_except, set_train_mode
from facial_recognition_system.training.loop import (
    EpochRecord,
    TrainingHistory,
    fit,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "ArcFaceHead",
    "EpochRecord",
    "FaceDataset",
    "TrainingHistory",
    "build_augmentation",
    "fit",
    "freeze_except",
    "label_indices",
    "load_checkpoint",
    "save_checkpoint",
    "select_training_subset",
    "set_train_mode",
]
