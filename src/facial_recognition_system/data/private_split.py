"""Open-set split of the private dataset (decision D9).

Session 1 of each gallery participant is the gallery (enrollment). Session 2 is
shuffled and split half to validation and half to test. Two participants stay
out of the gallery as unknowns, one in validation and one in test, with every
photo of both sessions. Paths are relative to the private crops folder
(``P01/session1/IMG_0001.png``) and every label is an anonymous ID or
``UNKNOWN_LABEL``, so no name reaches the split. The split is not versioned: it
is rebuilt at run time with the same seed.
"""

import random
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from facial_recognition_system.data.errors import (
    IncompleteParticipantError,
    NonAnonymousFolderError,
    NotEnoughParticipantsError,
)
from facial_recognition_system.evaluation.open_set import ANONYMOUS_ID_PATTERN, UNKNOWN_LABEL

SESSIONS: tuple[str, str] = ("session1", "session2")
PRIVATE_SPLIT_ORDER: tuple[str, str, str] = ("gallery", "val", "test")
# The gallery needs at least two people, or identification is trivial.
MIN_GALLERY_PARTICIPANTS: int = 2
# Session 2 must give at least one photo to validation and one to test.
MIN_SESSION2_PHOTOS: int = 2
CROP_PATTERN: re.Pattern[str] = re.compile(r"\.png$", re.IGNORECASE)

# Participant ID -> session folder -> crop paths, relative to the crops folder.
PhotosBySession = Mapping[str, Mapping[str, Sequence[str]]]


class PrivateSplitConfig(BaseModel):
    seed: int = 42
    # One unknown participant goes to validation and the other to test (D9).
    unknown_participants: int = Field(default=2, ge=2, le=2)


class LabeledPhoto(BaseModel):
    path: str
    label: str


class PrivateSplit(BaseModel):
    """Labeled photos of each part; unknowns are labeled ``UNKNOWN_LABEL``."""

    config: PrivateSplitConfig
    gallery_participants: list[str]
    gallery: list[LabeledPhoto]
    val: list[LabeledPhoto]
    test: list[LabeledPhoto]

    def paths_by_split(self) -> dict[str, list[str]]:
        return {
            split: [photo.path for photo in getattr(self, split)] for split in PRIVATE_SPLIT_ORDER
        }


def find_private_crops(crops_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Lists the crops of ``<crops_dir>/<ID>/session{1,2}/``, in a stable order.

    Returns:
        Crop paths relative to ``crops_dir`` by participant and session. A
        missing session folder gives an empty list, which the split rejects.
    """
    photos: dict[str, dict[str, list[str]]] = {}
    for participant_dir in sorted(path for path in crops_dir.iterdir() if path.is_dir()):
        photos[participant_dir.name] = {
            session: sorted(
                path.relative_to(crops_dir).as_posix()
                for path in (participant_dir / session).glob("*")
                if CROP_PATTERN.search(path.name)
            )
            for session in SESSIONS
        }
    return photos


def build_private_split(
    photos: PhotosBySession, config: PrivateSplitConfig | None = None
) -> PrivateSplit:
    """Splits the private crops into gallery, validation and test (D9).

    Args:
        photos: Crop paths by anonymous participant ID and session.
        config: Seed and number of unknown participants.

    Returns:
        The split, deterministic for the same photos and seed.

    Raises:
        NonAnonymousFolderError: A participant folder is not an ID like ``P01``.
        NotEnoughParticipantsError: Fewer participants than the gallery plus the unknowns.
        IncompleteParticipantError: A gallery participant has no session 1 photo or
            fewer than two session 2 photos.
    """
    config = config or PrivateSplitConfig()
    named = [participant for participant in photos if not ANONYMOUS_ID_PATTERN.match(participant)]
    if named:
        raise NonAnonymousFolderError(len(named))

    required = MIN_GALLERY_PARTICIPANTS + config.unknown_participants
    participants = sorted(photos)
    if len(participants) < required:
        raise NotEnoughParticipantsError(len(participants), required)

    rng = random.Random(config.seed)
    unknown_val, unknown_test = rng.sample(participants, config.unknown_participants)
    gallery_participants = [p for p in participants if p not in (unknown_val, unknown_test)]

    gallery: list[LabeledPhoto] = []
    val: list[LabeledPhoto] = []
    test: list[LabeledPhoto] = []
    for participant in gallery_participants:
        session1 = list(photos[participant].get("session1", []))
        session2 = sorted(photos[participant].get("session2", []))
        if not session1:
            raise IncompleteParticipantError(participant, "session1", 0, 1)
        if len(session2) < MIN_SESSION2_PHOTOS:
            raise IncompleteParticipantError(
                participant, "session2", len(session2), MIN_SESSION2_PHOTOS
            )

        rng.shuffle(session2)
        half = len(session2) // 2
        gallery += _labeled(session1, participant)
        val += _labeled(session2[:half], participant)
        test += _labeled(session2[half:], participant)

    val += _labeled(_all_photos(photos[unknown_val]), UNKNOWN_LABEL)
    test += _labeled(_all_photos(photos[unknown_test]), UNKNOWN_LABEL)

    return PrivateSplit(
        config=config,
        gallery_participants=gallery_participants,
        gallery=sorted(gallery, key=lambda photo: photo.path),
        val=sorted(val, key=lambda photo: photo.path),
        test=sorted(test, key=lambda photo: photo.path),
    )


def exclude_private_photos(split: PrivateSplit, excluded: Collection[str]) -> PrivateSplit:
    """Removes photos (the later copies found by the leakage check) from every part."""
    excluded_set = set(excluded)

    def keep(photos: list[LabeledPhoto]) -> list[LabeledPhoto]:
        return [photo for photo in photos if photo.path not in excluded_set]

    return split.model_copy(
        update={"gallery": keep(split.gallery), "val": keep(split.val), "test": keep(split.test)}
    )


def _labeled(paths: Sequence[str], label: str) -> list[LabeledPhoto]:
    return [LabeledPhoto(path=path, label=label) for path in paths]


def _all_photos(sessions: Mapping[str, Sequence[str]]) -> list[str]:
    return [path for session in SESSIONS for path in sessions.get(session, [])]
