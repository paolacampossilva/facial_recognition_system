"""Experiment configuration read from ``configs/<EXP>.yaml`` (decision D11)."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

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


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=128, ge=1)


class ExperimentConfig(BaseModel):
    """One experiment: what it tests, which model and data it uses and its seed."""

    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(pattern=r"^E\d{2}$")
    hypothesis: str
    strategy: str
    model: ModelConfig
    dataset: DatasetConfig = DatasetConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    seed: int = 42


def load_experiment_config(path: Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Reads and validates an experiment YAML.

    Returns:
        The validated config and the raw YAML mapping, which the results JSON
        stores as-is (the ``config`` field of the contract).
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw), raw
