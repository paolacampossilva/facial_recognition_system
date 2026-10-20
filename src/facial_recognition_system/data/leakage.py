"""Cross-split near-duplicate detection with difference hashes (decision D9).

A 64-bit dHash of the face crop is useless here: every aligned face shrinks to
almost the same 9x8 gradient, so different people collide. Detection runs in
two stages instead:

1. candidates: 64-bit dHash of the original photos, where the background and
   framing tell photos apart;
2. confirmation: 256-bit dHash of the face crops, which separates the same
   face from different faces with a similar background.

Of each confirmed pair, the copy in the later split (train < val < test) is
the one to remove, so validation and test never share a photo with an earlier
split.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from tqdm import tqdm

from facial_recognition_system.data.errors import UnknownSplitError


class LeakageConfig(BaseModel):
    # Candidate hashes are compared as 64-bit integers, so at most 8 x 8 bits.
    candidate_hash_size: int = Field(default=8, ge=2, le=8)
    candidate_max_distance: int = Field(default=4, ge=0)
    confirm_hash_size: int = Field(default=16, ge=2)
    confirm_max_distance: int = Field(default=48, ge=0)


class HashedPhoto(BaseModel):
    """A photo of a split with the hashes of both stages."""

    path: str
    split: str
    candidate_hash: int
    confirm_hash: int


class NearDuplicate(BaseModel):
    """Two photos of different splits that are the same image; ``path_b`` is removed."""

    path_a: str
    split_a: str
    path_b: str
    split_b: str
    candidate_distance: int
    confirm_distance: int


class LeakageReport(BaseModel):
    """What the check compared, what it found and what was removed."""

    config: LeakageConfig
    photos_per_split: dict[str, int]
    duplicates: list[NearDuplicate]
    excluded: list[str]


def dhash(image: Image.Image, hash_size: int) -> int:
    """Difference hash: one bit per horizontal neighbor pair of the grayscale thumbnail.

    Args:
        image: Image to hash.
        hash_size: Side of the hash; the result has ``hash_size ** 2`` bits.

    Returns:
        The hash as a non-negative integer.
    """
    thumbnail = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(thumbnail, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (hash_a ^ hash_b).bit_count()


def hash_photos(
    paths_by_split: Mapping[str, Sequence[str]],
    originals_dir: Path,
    crops_dir: Path,
    original_suffix: str,
    config: LeakageConfig,
) -> list[HashedPhoto]:
    """Hashes the original photo and the face crop of every listed path.

    Args:
        paths_by_split: Crop paths of each split, relative to ``crops_dir``.
        originals_dir: Folder of the original photos, with the same layout.
        crops_dir: Folder of the face crops.
        original_suffix: Extension of the original photos (``.jpg`` in LFW).
        config: Hash sizes of both stages.

    Returns:
        One hashed photo per path, in split order.
    """
    photos: list[HashedPhoto] = []
    for split, paths in paths_by_split.items():
        for path in tqdm(paths, desc=f"hashes {split}", unit="img"):
            original = _open(originals_dir / Path(path).with_suffix(original_suffix))
            crop = _open(crops_dir / path)
            photos.append(
                HashedPhoto(
                    path=path,
                    split=split,
                    candidate_hash=dhash(original, config.candidate_hash_size),
                    confirm_hash=dhash(crop, config.confirm_hash_size),
                )
            )
    return photos


def _open(path: Path) -> Image.Image:
    """Opens an image upright, applying the EXIF orientation of phone photos."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image)


def find_cross_split_duplicates(
    photos: Sequence[HashedPhoto], split_order: Sequence[str], config: LeakageConfig
) -> list[NearDuplicate]:
    """Finds photos that appear, as near-duplicates, in more than one split.

    Args:
        photos: Every photo with its split and hashes.
        split_order: Splits from first to last; the copy in the later split is
            ``path_b`` of the result.
        config: Hash sizes and Hamming distance limits of both stages.

    Returns:
        The confirmed near-duplicates, sorted by the removed path.

    Raises:
        UnknownSplitError: A photo belongs to a split missing from ``split_order``.
    """
    rank = {split: index for index, split in enumerate(split_order)}
    unknown_splits = {photo.split for photo in photos} - rank.keys()
    if unknown_splits:
        raise UnknownSplitError(sorted(unknown_splits))

    candidates = np.array([photo.candidate_hash for photo in photos], dtype=np.uint64)
    split_ranks = np.array([rank[photo.split] for photo in photos])

    duplicates: list[NearDuplicate] = []
    for index, photo in enumerate(photos):
        distances = np.bitwise_count(candidates ^ candidates[index])
        later_split = split_ranks > split_ranks[index]
        for other in np.flatnonzero(later_split & (distances <= config.candidate_max_distance)):
            match = photos[other]
            confirm_distance = hamming_distance(photo.confirm_hash, match.confirm_hash)
            if confirm_distance > config.confirm_max_distance:
                continue
            duplicates.append(
                NearDuplicate(
                    path_a=photo.path,
                    split_a=photo.split,
                    path_b=match.path,
                    split_b=match.split,
                    candidate_distance=int(distances[other]),
                    confirm_distance=confirm_distance,
                )
            )

    return sorted(duplicates, key=lambda duplicate: (duplicate.path_b, duplicate.path_a))


def check_leakage(
    photos: Sequence[HashedPhoto], split_order: Sequence[str], config: LeakageConfig
) -> LeakageReport:
    """Finds the cross-split near-duplicates and the later-split copies to remove."""
    duplicates = find_cross_split_duplicates(photos, split_order, config)
    photos_per_split = {
        split: sum(photo.split == split for photo in photos) for split in split_order
    }
    return LeakageReport(
        config=config,
        photos_per_split=photos_per_split,
        duplicates=duplicates,
        excluded=sorted({duplicate.path_b for duplicate in duplicates}),
    )
