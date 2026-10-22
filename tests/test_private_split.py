from pathlib import Path

import pytest

from facial_recognition_system.data import (
    IncompleteParticipantError,
    NonAnonymousFolderError,
    NotEnoughParticipantsError,
    build_private_split,
    exclude_private_photos,
    find_private_crops,
)
from facial_recognition_system.evaluation import UNKNOWN_LABEL


def _photos(participants: int, session1: int = 4, session2: int = 4) -> dict:
    return {
        f"P{index:02d}": {
            "session1": [f"P{index:02d}/session1/{photo}.png" for photo in range(session1)],
            "session2": [f"P{index:02d}/session2/{photo}.png" for photo in range(session2)],
        }
        for index in range(1, participants + 1)
    }


@pytest.mark.unit
def test_gallery_is_session1_and_session2_is_split_between_val_and_test() -> None:
    photos = _photos(6)

    split = build_private_split(photos)

    assert len(split.gallery_participants) == 4
    for participant in split.gallery_participants:
        gallery = {p.path for p in split.gallery if p.label == participant}
        val = {p.path for p in split.val if p.label == participant}
        test = {p.path for p in split.test if p.label == participant}
        assert gallery == set(photos[participant]["session1"])
        assert len(val) == len(test) == 2
        assert val | test == set(photos[participant]["session2"])


@pytest.mark.unit
def test_two_participants_are_unknown_one_in_val_and_one_in_test() -> None:
    photos = _photos(6)

    split = build_private_split(photos)

    unknowns = sorted(set(photos) - set(split.gallery_participants))
    assert len(unknowns) == 2
    val_unknown = {p.path.split("/")[0] for p in split.val if p.label == UNKNOWN_LABEL}
    test_unknown = {p.path.split("/")[0] for p in split.test if p.label == UNKNOWN_LABEL}
    assert len(val_unknown) == len(test_unknown) == 1
    assert val_unknown | test_unknown == set(unknowns)
    # Every photo of both sessions of an unknown goes to its split.
    assert sum(p.label == UNKNOWN_LABEL for p in split.val) == 8
    assert not any(p.path.split("/")[0] in unknowns for p in split.gallery)


@pytest.mark.unit
def test_split_is_deterministic_for_the_same_seed() -> None:
    assert build_private_split(_photos(6)) == build_private_split(_photos(6))


@pytest.mark.unit
def test_split_rejects_fewer_than_four_participants() -> None:
    with pytest.raises(NotEnoughParticipantsError):
        build_private_split(_photos(3))


@pytest.mark.unit
def test_split_rejects_named_folders_without_repeating_the_name() -> None:
    photos = _photos(5) | {"Maria_Silva": _photos(1)["P01"]}

    with pytest.raises(NonAnonymousFolderError) as error:
        build_private_split(photos)

    assert "Maria" not in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize(("session1", "session2"), [(0, 4), (4, 1)])
def test_split_rejects_gallery_participant_without_enough_photos(
    session1: int, session2: int
) -> None:
    photos = {participant: _photos(1, session1, session2)["P01"] for participant in _photos(6)}

    with pytest.raises(IncompleteParticipantError):
        build_private_split(photos)


@pytest.mark.unit
def test_exclude_private_photos_drops_them_from_every_part() -> None:
    split = build_private_split(_photos(6))
    removed = [split.val[0].path, split.gallery[0].path]

    result = exclude_private_photos(split, removed)

    remaining = {p.path for p in result.gallery + result.val + result.test}
    assert not remaining & set(removed)
    assert len(remaining) == len(split.gallery) + len(split.val) + len(split.test) - 2


@pytest.mark.unit
def test_find_private_crops_lists_png_crops_by_session(tmp_path: Path) -> None:
    for path in ("P01/session1/a.png", "P01/session2/b.PNG", "P01/session2/notes.txt"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(b"")

    assert find_private_crops(tmp_path) == {
        "P01": {"session1": ["P01/session1/a.png"], "session2": ["P01/session2/b.PNG"]}
    }
