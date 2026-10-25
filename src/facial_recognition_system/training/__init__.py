"""Partial fine-tuning of the FaceNet backbone (decisions D2 and D5)."""

from facial_recognition_system.training.arcface import ArcFaceHead, ArcFaceLoss
from facial_recognition_system.training.data import (
    FaceDataset,
    IdentityBatchSampler,
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
from facial_recognition_system.training.triplet import TripletLoss, batch_hard_triplet_loss

__all__ = [
    "ArcFaceHead",
    "ArcFaceLoss",
    "EpochRecord",
    "FaceDataset",
    "IdentityBatchSampler",
    "TrainingHistory",
    "TripletLoss",
    "batch_hard_triplet_loss",
    "build_augmentation",
    "fit",
    "freeze_except",
    "label_indices",
    "load_checkpoint",
    "save_checkpoint",
    "select_training_subset",
    "set_train_mode",
]
