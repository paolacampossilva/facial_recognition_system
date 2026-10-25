"""Experiment configuration read from ``configs/<EXP>.yaml`` (decision D11)."""

import math
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from facial_recognition_system.models import Backbone


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: Backbone
    pretrained: bool = True


class DatasetConfig(BaseModel):
    """Public split lists of the contract: ``<splits_dir>/lfw_<split>.txt``, relative to crops."""

    model_config = ConfigDict(extra="forbid")

    name: str = "lfw"
    crops_dir: Path = Path("data/public/lfw_crop160")
    splits_dir: Path = Path("data/splits")
    splits: list[str] = Field(default=["val", "test"], min_length=1)

    def split_file(self, split: str) -> Path:
        return self.splits_dir / f"{self.name}_{split}.txt"

    def pairs_file(self, split: str) -> Path:
        return self.splits_dir / f"{self.name}_{split}_pairs.csv"


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=128, ge=1)


class Loss(StrEnum):
    ARCFACE = "arcface"
    TRIPLET = "triplet"


class SubsetConfig(BaseModel):
    """Fine-tuning subset of the train split (decision D2)."""

    model_config = ConfigDict(extra="forbid")

    min_photos: int = Field(default=4, ge=1)
    max_photos: int = Field(default=40, ge=1)


class AugmentationConfig(BaseModel):
    """Light augmentation applied only to the training photos (decision D5)."""

    model_config = ConfigDict(extra="forbid")

    horizontal_flip: bool = True
    brightness: float = Field(default=0.2, ge=0)
    contrast: float = Field(default=0.2, ge=0)
    saturation: float = Field(default=0.2, ge=0)


class ArcFaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale: float = Field(default=32.0, gt=0)
    margin: float = Field(default=0.5, ge=0, lt=math.pi / 2)


class TripletConfig(BaseModel):
    """Batch-hard triplet loss over P x K batches (decision D5)."""

    model_config = ConfigDict(extra="forbid")

    margin: float = Field(default=0.2, gt=0)
    identities_per_batch: int = Field(default=32, ge=2)
    photos_per_identity: int = Field(default=4, ge=2)


class TrainingConfig(BaseModel):
    """Partial fine-tuning of decision D5, early stopped on the validation EER."""

    model_config = ConfigDict(extra="forbid")

    loss: Loss = Loss.ARCFACE
    # Top-level children of the backbone network that train; everything else is frozen.
    trainable: list[str] = Field(default=["block8", "last_linear", "last_bn"], min_length=1)
    subset: SubsetConfig = SubsetConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    arcface: ArcFaceConfig = ArcFaceConfig()
    triplet: TripletConfig = TripletConfig()
    batch_size: int = Field(default=128, ge=2)
    max_epochs: int = Field(default=30, ge=1)
    patience: int = Field(default=5, ge=1)
    backbone_lr: float = Field(default=1e-4, gt=0)
    head_lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=5e-4, ge=0)
    num_workers: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_triplet_batches(self) -> Self:
        if self.loss is not Loss.TRIPLET:
            return self
        identities, photos = self.triplet.identities_per_batch, self.triplet.photos_per_identity
        if self.batch_size != identities * photos:
            raise ValueError(
                f"triplet batches have {identities} x {photos} photos, "
                f"so batch_size must be {identities * photos}"
            )
        if self.subset.min_photos < photos:
            raise ValueError("subset.min_photos must be at least triplet.photos_per_identity")
        return self


class ExperimentConfig(BaseModel):
    """One experiment: what it tests, which model and data it uses and its seed."""

    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(pattern=r"^E\d{2}$")
    hypothesis: str
    strategy: str
    model: ModelConfig
    dataset: DatasetConfig = DatasetConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    # Absent in experiments without fine-tuning (E01, E02).
    training: TrainingConfig | None = None
    seed: int = 42


def load_experiment_config(path: Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Reads and validates an experiment YAML.

    Returns:
        The validated config and the raw YAML mapping, which the results JSON
        stores as-is (the ``config`` field of the contract).
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw), raw
