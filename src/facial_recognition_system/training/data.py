"""Fine-tuning subset (decision D2) and the training dataset with augmentation (decision D5)."""

import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from facial_recognition_system.experiments import AugmentationConfig, identity_of
from facial_recognition_system.preprocessing import load_image

Augmentation = Callable[[Image.Image], Image.Image]


def select_training_subset(
    paths: Sequence[str], min_photos: int, max_photos: int, seed: int
) -> list[str]:
    """Keeps identities with at least ``min_photos`` photos, at most ``max_photos`` each.

    Identities with more photos than the cap get a random sample, so a few
    people (George W. Bush has 530 photos) do not dominate the training. The
    same input and seed always give the same subset.
    """
    photos_by_identity: defaultdict[str, list[str]] = defaultdict(list)
    for path in paths:
        photos_by_identity[identity_of(path)].append(path)

    rng = random.Random(seed)
    selected: list[str] = []
    for identity in sorted(photos_by_identity):
        photos = sorted(photos_by_identity[identity])
        if len(photos) < min_photos:
            continue
        if len(photos) > max_photos:
            photos = sorted(rng.sample(photos, max_photos))
        selected.extend(photos)
    return selected


def label_indices(paths: Sequence[str]) -> dict[str, int]:
    """Maps each identity to a class index of the classification head, in sorted order."""
    identities = sorted({identity_of(path) for path in paths})
    return {identity: index for index, identity in enumerate(identities)}


def build_augmentation(config: AugmentationConfig) -> Augmentation | None:
    """Random flip and light color jitter, or None when nothing is enabled."""
    steps: list[Augmentation] = []
    if config.horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    if config.brightness or config.contrast or config.saturation:
        steps.append(
            transforms.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
            )
        )
    return transforms.Compose(steps) if steps else None


class FaceDataset(Dataset[tuple[torch.Tensor, int]]):
    """Face crops with their class index, preprocessed for the backbone.

    Only the training dataset gets an ``augmentation``; validation and test go
    through ``embed_crops``, which never augments (section 30).
    """

    def __init__(
        self,
        crops_dir: Path,
        paths: Sequence[str],
        labels: Mapping[str, int],
        preprocess: Callable[[Image.Image], torch.Tensor],
        augmentation: Augmentation | None = None,
    ) -> None:
        self.crops_dir = crops_dir
        self.paths = list(paths)
        self.labels = labels
        self.preprocess = preprocess
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        face = load_image(self.crops_dir / path)
        if self.augmentation is not None:
            face = self.augmentation(face)
        return self.preprocess(face), self.labels[identity_of(path)]
