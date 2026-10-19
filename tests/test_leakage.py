from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from facial_recognition_system.data import (
    HashedPhoto,
    LeakageConfig,
    UnknownSplitError,
    check_leakage,
    dhash,
    find_cross_split_duplicates,
    hamming_distance,
    hash_photos,
)

CONFIG = LeakageConfig()
SPLIT_ORDER = ("train", "val", "test")


def _noise(seed: int, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def _photo(path: str, split: str, candidate: int, confirm: int) -> HashedPhoto:
    return HashedPhoto(path=path, split=split, candidate_hash=candidate, confirm_hash=confirm)


@pytest.mark.unit
def test_dhash_of_increasing_and_decreasing_gradients() -> None:
    increasing = Image.fromarray(np.tile(np.arange(0, 250, 10, dtype=np.uint8), (25, 1)))
    decreasing = increasing.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    assert dhash(increasing, 8) == 2**64 - 1
    assert dhash(decreasing, 8) == 0


@pytest.mark.unit
def test_dhash_is_stable_for_the_same_image_and_differs_for_another() -> None:
    image = _noise(1)

    assert dhash(image, 16) == dhash(image.copy(), 16)
    assert hamming_distance(dhash(image, 16), dhash(_noise(2), 16)) > 48


@pytest.mark.unit
def test_hamming_distance_counts_different_bits() -> None:
    assert hamming_distance(0b1011, 0b0010) == 2
    assert hamming_distance(5, 5) == 0


@pytest.mark.unit
def test_confirmed_duplicate_marks_the_copy_in_the_later_split() -> None:
    photos = [
        _photo("Ada/Ada_0001.png", "test", candidate=0b1111, confirm=0),
        _photo("Bob/Bob_0001.png", "train", candidate=0b1110, confirm=1),
    ]

    duplicates = find_cross_split_duplicates(photos, SPLIT_ORDER, CONFIG)

    assert len(duplicates) == 1
    assert (duplicates[0].split_a, duplicates[0].path_a) == ("train", "Bob/Bob_0001.png")
    assert (duplicates[0].split_b, duplicates[0].path_b) == ("test", "Ada/Ada_0001.png")
    assert (duplicates[0].candidate_distance, duplicates[0].confirm_distance) == (1, 1)


@pytest.mark.unit
def test_candidate_without_confirmation_is_not_a_duplicate() -> None:
    # Similar backgrounds (candidate stage) but different faces (confirmation stage).
    photos = [
        _photo("Ada/Ada_0001.png", "train", candidate=0, confirm=0),
        _photo("Bob/Bob_0001.png", "val", candidate=0, confirm=2**60 - 1),
    ]

    assert find_cross_split_duplicates(photos, SPLIT_ORDER, CONFIG) == []


@pytest.mark.unit
def test_far_candidates_are_not_duplicates() -> None:
    photos = [
        _photo("Ada/Ada_0001.png", "train", candidate=0, confirm=0),
        _photo("Bob/Bob_0001.png", "val", candidate=0b11111, confirm=0),
    ]

    assert find_cross_split_duplicates(photos, SPLIT_ORDER, CONFIG) == []


@pytest.mark.unit
def test_duplicates_inside_the_same_split_are_ignored() -> None:
    photos = [
        _photo("Ada/Ada_0001.png", "val", candidate=7, confirm=7),
        _photo("Ada/Ada_0002.png", "val", candidate=7, confirm=7),
    ]

    assert find_cross_split_duplicates(photos, SPLIT_ORDER, CONFIG) == []


@pytest.mark.unit
def test_check_leakage_excludes_each_later_copy_once() -> None:
    photos = [
        _photo("Ada/Ada_0001.png", "train", candidate=1, confirm=1),
        _photo("Bob/Bob_0001.png", "val", candidate=1, confirm=1),
        _photo("Cid/Cid_0001.png", "test", candidate=1, confirm=1),
        _photo("Dan/Dan_0001.png", "test", candidate=0xFFFF_FFFF_0000_0000, confirm=0),
    ]

    report = check_leakage(photos, SPLIT_ORDER, CONFIG)

    assert len(report.duplicates) == 3
    assert report.excluded == ["Bob/Bob_0001.png", "Cid/Cid_0001.png"]
    assert report.photos_per_split == {"train": 1, "val": 1, "test": 2}


@pytest.mark.unit
def test_photo_in_unlisted_split_raises() -> None:
    photos = [_photo("Ada/Ada_0001.png", "session3", candidate=0, confirm=0)]

    with pytest.raises(UnknownSplitError, match="session3"):
        find_cross_split_duplicates(photos, SPLIT_ORDER, CONFIG)


@pytest.mark.unit
def test_candidate_hash_larger_than_64_bits_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LeakageConfig(candidate_hash_size=9)


@pytest.mark.unit
def test_hash_photos_finds_the_same_photo_under_two_identities(tmp_path: Path) -> None:
    originals, crops = tmp_path / "originals", tmp_path / "crops"
    images = {"Ada/Ada_0001": _noise(1), "Bob/Bob_0001": _noise(1), "Cid/Cid_0001": _noise(3)}
    for name, image in images.items():
        (originals / name).parent.mkdir(parents=True, exist_ok=True)
        (crops / name).parent.mkdir(parents=True, exist_ok=True)
        image.save(originals / f"{name}.jpg", quality=95)
        image.resize((32, 32)).save(crops / f"{name}.png")
    paths_by_split = {
        "train": ["Ada/Ada_0001.png"],
        "val": ["Bob/Bob_0001.png"],
        "test": ["Cid/Cid_0001.png"],
    }

    photos = hash_photos(paths_by_split, originals, crops, ".jpg", CONFIG)
    report = check_leakage(photos, SPLIT_ORDER, CONFIG)

    assert [photo.split for photo in photos] == ["train", "val", "test"]
    assert report.excluded == ["Bob/Bob_0001.png"]
