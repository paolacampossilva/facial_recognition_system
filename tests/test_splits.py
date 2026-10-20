import pytest
from pydantic import ValidationError

from facial_recognition_system.data import (
    LfwSplits,
    NotEnoughNegativePairsError,
    NotEnoughUnknownsError,
    SplitConfig,
    build_lfw_splits,
    exclude_photos,
)

CONFIG = SplitConfig(unknowns_per_split=10)


def _identity(path: str) -> str:
    return path.rsplit("/", 1)[0]


def _dataset(photo_counts: dict[str, int]) -> dict[str, list[str]]:
    return {
        name: [f"{name}/{name}_{index:04d}.png" for index in range(1, count + 1)]
        for name, count in photo_counts.items()
    }


@pytest.fixture
def photos() -> dict[str, list[str]]:
    counts = {f"Few_{index:02d}": 2 + index % 3 for index in range(40)}
    counts |= {f"Some_{index:02d}": 5 + index % 5 for index in range(20)}
    counts |= {f"Many_{index:02d}": 10 + index for index in range(20)}
    counts |= {f"Single_{index:02d}": 1 for index in range(30)}
    return _dataset(counts)


@pytest.fixture
def splits(photos: dict[str, list[str]]) -> LfwSplits:
    return build_lfw_splits(photos, CONFIG)


@pytest.mark.unit
def test_no_identity_appears_in_more_than_one_split(splits: LfwSplits) -> None:
    train = {_identity(path) for path in splits.train}
    val = {_identity(path) for path in splits.val}
    test = {_identity(path) for path in splits.test}

    assert not train & val
    assert not train & test
    assert not val & test


@pytest.mark.unit
def test_every_photo_lands_in_exactly_one_split(
    photos: dict[str, list[str]], splits: LfwSplits
) -> None:
    all_photos = sorted(path for paths in photos.values() for path in paths)

    assert sorted(splits.train + splits.val + splits.test) == all_photos


@pytest.mark.unit
def test_each_stratum_is_split_70_15_15(splits: LfwSplits) -> None:
    many_train = {_identity(path) for path in splits.train if path.startswith("Many_")}
    many_val = {_identity(path) for path in splits.val if path.startswith("Many_")}
    many_test = {_identity(path) for path in splits.test if path.startswith("Many_")}

    assert (len(many_train), len(many_val), len(many_test)) == (14, 3, 3)


@pytest.mark.unit
def test_single_photo_identities_go_half_to_val_and_half_to_test(splits: LfwSplits) -> None:
    val_singles = [path for path in splits.val if path.startswith("Single_")]
    test_singles = [path for path in splits.test if path.startswith("Single_")]

    assert not any(path.startswith("Single_") for path in splits.train)
    assert len(val_singles) == len(test_singles) == 15


@pytest.mark.unit
@pytest.mark.parametrize("split_name", ["val", "test"])
def test_pairs_are_balanced_and_use_only_photos_of_their_split(
    splits: LfwSplits, split_name: str
) -> None:
    photos = set(getattr(splits, split_name))
    pairs = getattr(splits, f"{split_name}_pairs")
    positives = [pair for pair in pairs if pair.same]
    negatives = [pair for pair in pairs if not pair.same]

    assert len(positives) == len(negatives) > 0
    assert all(pair.path_a in photos and pair.path_b in photos for pair in pairs)
    assert all(_identity(pair.path_a) == _identity(pair.path_b) for pair in positives)
    assert all(_identity(pair.path_a) != _identity(pair.path_b) for pair in negatives)


@pytest.mark.unit
@pytest.mark.parametrize("split_name", ["val", "test"])
def test_positive_pairs_are_all_combinations_up_to_the_cap(
    splits: LfwSplits, split_name: str
) -> None:
    cap = CONFIG.max_positive_pairs_per_identity
    photos = getattr(splits, split_name)
    pairs = getattr(splits, f"{split_name}_pairs")

    photo_counts: dict[str, int] = {}
    for path in photos:
        photo_counts[_identity(path)] = photo_counts.get(_identity(path), 0) + 1
    pair_counts: dict[str, int] = {}
    for pair in pairs:
        if pair.same:
            pair_counts[_identity(pair.path_a)] = pair_counts.get(_identity(pair.path_a), 0) + 1

    for identity, count in photo_counts.items():
        expected = min(count * (count - 1) // 2, cap)
        assert pair_counts.get(identity, 0) == expected


@pytest.mark.unit
@pytest.mark.parametrize("split_name", ["val", "test"])
def test_pairs_are_not_repeated(splits: LfwSplits, split_name: str) -> None:
    pairs = getattr(splits, f"{split_name}_pairs")
    keys = [tuple(sorted((pair.path_a, pair.path_b))) for pair in pairs]

    assert len(keys) == len(set(keys))


@pytest.mark.unit
def test_unknowns_are_single_photo_identities_of_their_split(splits: LfwSplits) -> None:
    assert len(splits.val_unknowns) == len(splits.test_unknowns) == CONFIG.unknowns_per_split
    assert set(splits.val_unknowns) <= set(splits.val)
    assert set(splits.test_unknowns) <= set(splits.test)
    assert all(path.startswith("Single_") for path in splits.val_unknowns + splits.test_unknowns)


@pytest.mark.unit
def test_same_seed_gives_same_splits_and_another_seed_changes_them(
    photos: dict[str, list[str]], splits: LfwSplits
) -> None:
    shuffled_input = {name: list(reversed(paths)) for name, paths in reversed(photos.items())}
    other_seed = SplitConfig(seed=7, unknowns_per_split=CONFIG.unknowns_per_split)

    assert build_lfw_splits(shuffled_input, CONFIG) == splits
    assert build_lfw_splits(photos, other_seed).test != splits.test


@pytest.mark.unit
def test_excluded_photos_leave_every_list_pair_and_unknown(splits: LfwSplits) -> None:
    in_pair = splits.test_pairs[0].path_a
    unknown = splits.val_unknowns[0]

    filtered = exclude_photos(splits, {in_pair, unknown})

    kept_paths = filtered.train + filtered.val + filtered.test
    kept_paths += filtered.val_unknowns + filtered.test_unknowns
    kept_pairs = filtered.val_pairs + filtered.test_pairs
    assert in_pair not in kept_paths and unknown not in kept_paths
    assert all(in_pair not in (pair.path_a, pair.path_b) for pair in kept_pairs)
    assert len(filtered.test) == len(splits.test) - 1
    assert len(filtered.val_unknowns) == len(splits.val_unknowns) - 1


@pytest.mark.unit
def test_excluding_nothing_keeps_the_splits(splits: LfwSplits) -> None:
    assert exclude_photos(splits, set()) == splits


@pytest.mark.unit
def test_too_few_single_photo_identities_raise(photos: dict[str, list[str]]) -> None:
    config = SplitConfig(unknowns_per_split=16)

    with pytest.raises(NotEnoughUnknownsError, match="15 single-photo identities"):
        build_lfw_splits(photos, config)


@pytest.mark.unit
def test_split_that_cannot_balance_its_positive_pairs_raises() -> None:
    # Val and test get one identity each with 2 photos, so no negative pair exists.
    photos = _dataset({f"Pair_{index}": 2 for index in range(4)})
    config = SplitConfig(train_fraction=0.5, val_fraction=0.25, unknowns_per_split=0)

    with pytest.raises(NotEnoughNegativePairsError):
        build_lfw_splits(photos, config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"train_fraction": 0.8, "val_fraction": 0.2},
        {"strata_lower_bounds": (1, 5)},
        {"strata_lower_bounds": (2, 10, 5)},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SplitConfig.model_validate(overrides)
