"""Fine-tuning subset (decision D2) and the training dataset with augmentation (decision D5)."""

import random
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
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


class IdentityBatchSampler(Sampler[list[int]]):
    """P x K batches for the triplet loss: P different people with K photos each.

    Every epoch, each identity's photos are shuffled and cut into groups of K
    (the leftover photos wait for a later epoch). Each batch then takes one
    group from P different identities, chosen at random with a weight equal to
    how many groups they still have, so people with many photos are spread over
    the epoch instead of piling up at the end. An epoch ends when fewer than P
    identities have groups left, which covers almost every photo: about as many
    batches as the shuffled loader of the ArcFace run.

    Args:
        labels: Class index of each photo of the dataset, in dataset order.
        identities_per_batch: P.
        photos_per_identity: K; identities with fewer photos never appear.
        seed: Seed of the shuffling; each epoch continues the same generator.
    """

    def __init__(
        self,
        labels: Sequence[int],
        identities_per_batch: int,
        photos_per_identity: int,
        seed: int,
    ) -> None:
        self.identities_per_batch = identities_per_batch
        self.photos_per_identity = photos_per_identity
        self._rng = random.Random(seed)
        indices_by_label: defaultdict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            indices_by_label[label].append(index)
        self._indices_by_label = {
            label: indices
            for label, indices in sorted(indices_by_label.items())
            if len(indices) >= photos_per_identity
        }
        if len(self._indices_by_label) < identities_per_batch:
            raise ValueError(
                f"{len(self._indices_by_label)} identities with at least "
                f"{photos_per_identity} photos, fewer than the {identities_per_batch} per batch"
            )

    def __iter__(self) -> Iterator[list[int]]:
        k = self.photos_per_identity
        groups: dict[int, list[list[int]]] = {}
        for label, indices in self._indices_by_label.items():
            shuffled = self._rng.sample(indices, len(indices))
            groups[label] = [shuffled[i : i + k] for i in range(0, len(shuffled) - k + 1, k)]

        while len(groups) >= self.identities_per_batch:
            # Weighted sampling without replacement (Efraimidis and Spirakis).
            keys = {label: self._rng.random() ** (1 / len(left)) for label, left in groups.items()}
            chosen = sorted(groups, key=keys.__getitem__, reverse=True)[: self.identities_per_batch]
            batch: list[int] = []
            for label in chosen:
                batch.extend(groups[label].pop())
                if not groups[label]:
                    del groups[label]
            yield batch
