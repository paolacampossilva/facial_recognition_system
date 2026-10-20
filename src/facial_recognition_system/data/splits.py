"""Identity-disjoint LFW splits, verification pairs and extra unknowns.

Identities with two or more photos are split into train, validation and test,
stratified by photo count. Single-photo identities go half to validation and
half to test, where they act as impostors in negative pairs and as extra
unknowns for the private open-set evaluation. No identity appears in more than
one split. Paths are relative to the crops folder, with the identity as the
parent folder (``George_W_Bush/George_W_Bush_0001.png``).
"""

import itertools
import random
from collections.abc import Collection, Mapping, Sequence
from typing import Self

from pydantic import BaseModel, Field, model_validator

from facial_recognition_system.data.errors import (
    NotEnoughNegativePairsError,
    NotEnoughUnknownsError,
)

PhotosByIdentity = Mapping[str, Sequence[str]]


class SplitConfig(BaseModel):
    """Parameters of the public split (protocol in docs/division.md, decision D9)."""

    seed: int = 42
    train_fraction: float = Field(default=0.70, gt=0, lt=1)
    val_fraction: float = Field(default=0.15, gt=0, lt=1)
    # Lower bound of each photo-count stratum: 2-4, 5-9 and 10+ photos.
    strata_lower_bounds: tuple[int, ...] = (2, 5, 10)
    # Caps identities with many photos so they do not dominate the positive pairs.
    max_positive_pairs_per_identity: int = Field(default=20, ge=1)
    unknowns_per_split: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def _check_fractions_and_strata(self) -> Self:
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("train_fraction + val_fraction must leave room for the test split")
        bounds = self.strata_lower_bounds
        if not bounds or bounds[0] != 2 or list(bounds) != sorted(set(bounds)):
            raise ValueError("strata_lower_bounds must start at 2 and be strictly increasing")
        return self


class ImagePair(BaseModel):
    path_a: str
    path_b: str
    same: bool


class LfwSplits(BaseModel):
    """Photo lists, verification pairs and extra unknowns of every split."""

    config: SplitConfig
    train: list[str]
    val: list[str]
    test: list[str]
    val_pairs: list[ImagePair]
    test_pairs: list[ImagePair]
    val_unknowns: list[str]
    test_unknowns: list[str]


def build_lfw_splits(
    photos_by_identity: PhotosByIdentity, config: SplitConfig | None = None
) -> LfwSplits:
    """Splits the dataset by identity and samples the pairs and unknowns.

    The same config and input always give the same result: every random choice
    comes from one generator seeded with ``config.seed``, over sorted inputs.

    Args:
        photos_by_identity: Relative photo paths of each identity.
        config: Split parameters. Defaults to the protocol values.

    Returns:
        The photo lists, pairs and unknowns of train, validation and test.

    Raises:
        NotEnoughUnknownsError: A split has too few single-photo identities.
        NotEnoughNegativePairsError: A split cannot balance its positive pairs.
    """
    config = config or SplitConfig()
    rng = random.Random(config.seed)
    photos = {identity: sorted(paths) for identity, paths in photos_by_identity.items()}

    train_ids, val_ids, test_ids = _split_multi_photo_identities(photos, config, rng)
    val_singletons, test_singletons = _split_single_photo_identities(photos, rng)

    val_ids += val_singletons
    test_ids += test_singletons

    return LfwSplits(
        config=config,
        train=_photos_of(train_ids, photos),
        val=_photos_of(val_ids, photos),
        test=_photos_of(test_ids, photos),
        val_pairs=_make_pairs("val", val_ids, photos, config, rng),
        test_pairs=_make_pairs("test", test_ids, photos, config, rng),
        val_unknowns=_sample_unknowns("val", val_singletons, photos, config, rng),
        test_unknowns=_sample_unknowns("test", test_singletons, photos, config, rng),
    )


def exclude_photos(splits: LfwSplits, excluded: Collection[str]) -> LfwSplits:
    """Removes photos from every list, with every pair and unknown that uses them.

    Applied after the split, so the rest of the sampling stays the same and the
    embeddings of the remaining photos keep valid; the pairs lose their balance
    only by the pairs removed.

    Args:
        splits: Splits as built by ``build_lfw_splits``.
        excluded: Relative paths to remove, such as near-duplicates across splits.

    Returns:
        A copy of ``splits`` without the excluded photos.
    """
    excluded_set = set(excluded)

    def keep_photos(paths: list[str]) -> list[str]:
        return [path for path in paths if path not in excluded_set]

    def keep_pairs(pairs: list[ImagePair]) -> list[ImagePair]:
        return [
            pair
            for pair in pairs
            if pair.path_a not in excluded_set and pair.path_b not in excluded_set
        ]

    return splits.model_copy(
        update={
            "train": keep_photos(splits.train),
            "val": keep_photos(splits.val),
            "test": keep_photos(splits.test),
            "val_pairs": keep_pairs(splits.val_pairs),
            "test_pairs": keep_pairs(splits.test_pairs),
            "val_unknowns": keep_photos(splits.val_unknowns),
            "test_unknowns": keep_photos(splits.test_unknowns),
        }
    )


def _split_multi_photo_identities(
    photos: Mapping[str, list[str]], config: SplitConfig, rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    """Splits identities with 2+ photos into train, val and test within each stratum."""
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for stratum in _strata(photos, config.strata_lower_bounds):
        rng.shuffle(stratum)
        train_end = round(len(stratum) * config.train_fraction)
        val_end = train_end + round(len(stratum) * config.val_fraction)
        train += stratum[:train_end]
        val += stratum[train_end:val_end]
        test += stratum[val_end:]

    return train, val, test


def _strata(photos: Mapping[str, list[str]], lower_bounds: Sequence[int]) -> list[list[str]]:
    """Groups identities with 2+ photos by photo count, each group sorted by name."""
    upper_bounds = [*lower_bounds[1:], None]
    strata: list[list[str]] = []
    for lower, upper in zip(lower_bounds, upper_bounds, strict=True):
        stratum = sorted(
            identity
            for identity, paths in photos.items()
            if len(paths) >= lower and (upper is None or len(paths) < upper)
        )
        strata.append(stratum)
    return strata


def _split_single_photo_identities(
    photos: Mapping[str, list[str]], rng: random.Random
) -> tuple[list[str], list[str]]:
    """Sends half of the single-photo identities to val and the rest to test."""
    singletons = sorted(identity for identity, paths in photos.items() if len(paths) == 1)
    rng.shuffle(singletons)
    half = len(singletons) // 2
    return singletons[:half], singletons[half:]


def _photos_of(identities: Sequence[str], photos: Mapping[str, list[str]]) -> list[str]:
    return sorted(path for identity in identities for path in photos[identity])


def _make_pairs(
    split: str,
    identities: Sequence[str],
    photos: Mapping[str, list[str]],
    config: SplitConfig,
    rng: random.Random,
) -> list[ImagePair]:
    """Builds every capped positive pair of the split plus as many negative pairs."""
    positives = _positive_pairs(identities, photos, config.max_positive_pairs_per_identity, rng)
    negatives = _negative_pairs(split, identities, photos, len(positives), rng)
    return positives + negatives


def _positive_pairs(
    identities: Sequence[str],
    photos: Mapping[str, list[str]],
    max_per_identity: int,
    rng: random.Random,
) -> list[ImagePair]:
    pairs: list[ImagePair] = []
    for identity in sorted(identities):
        combinations = list(itertools.combinations(photos[identity], 2))
        if len(combinations) > max_per_identity:
            combinations = sorted(rng.sample(combinations, max_per_identity))
        pairs += [ImagePair(path_a=a, path_b=b, same=True) for a, b in combinations]
    return pairs


def _negative_pairs(
    split: str,
    identities: Sequence[str],
    photos: Mapping[str, list[str]],
    count: int,
    rng: random.Random,
) -> list[ImagePair]:
    """Samples distinct pairs of photos from different identities.

    The identities are drawn first and the photos second, so an identity with
    hundreds of photos is as likely to appear as one with a single photo.
    """
    photo_total = sum(len(photos[identity]) for identity in identities)
    same_identity_pairs = sum(_pair_count(len(photos[identity])) for identity in identities)
    available = _pair_count(photo_total) - same_identity_pairs
    if available < count:
        raise NotEnoughNegativePairsError(split, available, count)

    sorted_identities = sorted(identities)
    seen: set[tuple[str, str]] = set()
    pairs: list[ImagePair] = []
    while len(pairs) < count:
        identity_a, identity_b = rng.sample(sorted_identities, 2)
        path_a, path_b = sorted((rng.choice(photos[identity_a]), rng.choice(photos[identity_b])))
        if (path_a, path_b) in seen:
            continue
        seen.add((path_a, path_b))
        pairs.append(ImagePair(path_a=path_a, path_b=path_b, same=False))
    return pairs


def _pair_count(size: int) -> int:
    return size * (size - 1) // 2


def _sample_unknowns(
    split: str,
    singletons: Sequence[str],
    photos: Mapping[str, list[str]],
    config: SplitConfig,
    rng: random.Random,
) -> list[str]:
    """Samples the extra unknowns of the private open-set evaluation."""
    if len(singletons) < config.unknowns_per_split:
        raise NotEnoughUnknownsError(split, len(singletons), config.unknowns_per_split)
    chosen = rng.sample(sorted(singletons), config.unknowns_per_split)
    return sorted(photos[identity][0] for identity in chosen)
