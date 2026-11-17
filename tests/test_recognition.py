"""Recognition tests (specs 003 and 004): scripted detector and embedder, real database."""

import logging
from pathlib import Path

import numpy as np
import pytest

from app.core.enrollment import EnrollmentService
from app.core.errors import EmptyGalleryError
from app.core.models import RecognitionStatus
from app.core.recognition import MAX_FACES, RecognitionService, decide
from app.storage.repository import SqlitePersonRepository
from facial_recognition_system.preprocessing import InvalidImageError, NoFaceDetectedError
from tests.app_support import (
    ANA,
    ANA_COLOR,
    BIA_COLOR,
    CARLA_COLOR,
    MIN_FACE_PX,
    NO_FACE,
    THRESHOLD,
    VECTORS,
    Color,
    build_encoder,
    group_photo,
    only_face,
    photo,
    request,
    with_cosine,
)

pytestmark = pytest.mark.unit

QUERY_COLOR: Color = (10, 10, 10)
QUERY = with_cosine(ANA, 0.8, seed=5)  # a face 0.8 similar to Ana's
WITH_QUERY = {**VECTORS, QUERY_COLOR: QUERY}


def recognizer(
    repository: SqlitePersonRepository,
    threshold: float = THRESHOLD,
    model_id: str = "fake-model",
) -> RecognitionService:
    return RecognitionService(
        build_encoder(WITH_QUERY, model_id), repository, threshold, MIN_FACE_PX
    )


def enroll(
    repository: SqlitePersonRepository, name: str, color: Color, model_id: str = "fake-model"
) -> None:
    encoder = build_encoder(WITH_QUERY, model_id)
    EnrollmentService(encoder, repository, THRESHOLD).enroll(request(name, photo(color)))


def stored_similarity(repository: SqlitePersonRepository) -> float:
    """The similarity of QUERY with the stored Ana, computed the way the strategy does."""
    gallery = repository.gallery("fake-model")
    return float((QUERY[None, :] @ gallery.embeddings.T)[0, 0])


def test_the_photo_of_an_enrolled_person_is_recognized(repository: SqlitePersonRepository) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    result = only_face(recognizer(repository).recognize(photo(ANA_COLOR)))

    assert result.status is RecognitionStatus.RECOGNIZED
    assert result.person_name == "Ana"
    assert result.similarity >= THRESHOLD


def test_the_closest_person_is_the_one_returned(repository: SqlitePersonRepository) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    enroll(repository, "Bia", BIA_COLOR)

    assert only_face(recognizer(repository).recognize(photo(BIA_COLOR))).person_name == "Bia"
    assert only_face(recognizer(repository).recognize(photo(ANA_COLOR))).person_name == "Ana"


def test_a_similarity_equal_to_the_threshold_is_recognized(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    similarity = stored_similarity(repository)

    result = only_face(recognizer(repository, threshold=similarity).recognize(photo(QUERY_COLOR)))

    assert result.status is RecognitionStatus.RECOGNIZED
    assert result.similarity == similarity


def test_a_similarity_just_below_the_threshold_is_unknown(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    threshold = float(np.nextafter(stored_similarity(repository), 1.0))

    result = only_face(recognizer(repository, threshold=threshold).recognize(photo(QUERY_COLOR)))

    assert result.status is RecognitionStatus.UNKNOWN


@pytest.mark.parametrize(
    ("threshold", "status"),
    [(0.7, RecognitionStatus.RECOGNIZED), (0.9, RecognitionStatus.UNKNOWN)],
)
def test_the_threshold_comes_from_the_configuration_and_changes_the_result(
    repository: SqlitePersonRepository, threshold: float, status: RecognitionStatus
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    result = only_face(
        recognizer(repository, threshold=threshold).recognize(photo(QUERY_COLOR))  # 0.8
    )

    assert result.status is status


def test_only_embeddings_of_the_configured_model_are_compared(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR, model_id="model-a")
    enroll(repository, "Bia", BIA_COLOR, model_id="model-b")

    only_a = only_face(recognizer(repository, model_id="model-a").recognize(photo(BIA_COLOR)))
    only_b = only_face(recognizer(repository, model_id="model-b").recognize(photo(BIA_COLOR)))

    assert only_a.status is RecognitionStatus.UNKNOWN  # Bia's own vectors are another model's
    assert only_b.person_name == "Bia"


def test_an_empty_gallery_is_an_error_and_not_an_unknown_person(
    repository: SqlitePersonRepository,
) -> None:
    with pytest.raises(EmptyGalleryError):
        recognizer(repository).recognize(photo(ANA_COLOR))


def test_a_gallery_with_only_another_model_counts_as_empty(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR, model_id="model-a")

    with pytest.raises(EmptyGalleryError):
        recognizer(repository, model_id="model-b").recognize(photo(ANA_COLOR))


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (photo(width=NO_FACE), NoFaceDetectedError),
        (group_photo((ANA_COLOR, 50)), NoFaceDetectedError),  # the only face is too small
        (b"not an image", InvalidImageError),
    ],
)
def test_image_errors_propagate_and_vary_with_the_faces_found(
    repository: SqlitePersonRepository, image: bytes, error: type[Exception]
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    with pytest.raises(error):
        recognizer(repository).recognize(image)


def test_recognizing_changes_nothing_in_the_database_or_on_disk(tmp_path: Path) -> None:
    folder = tmp_path / "private"
    repository = SqlitePersonRepository(folder / "app.db")
    enroll(repository, "Ana", ANA_COLOR)
    database = folder / "app.db"
    before = (repository.list_persons(), database.read_bytes(), sorted(folder.iterdir()))

    recognizer(repository).recognize(photo(ANA_COLOR))
    recognizer(repository).recognize(photo(BIA_COLOR))

    after = (repository.list_persons(), database.read_bytes(), sorted(folder.iterdir()))
    repository.close()
    assert after == before


def test_an_enrollment_and_a_deletion_count_at_once_without_restarting(
    repository: SqlitePersonRepository,
) -> None:
    service = recognizer(repository)  # the same service the whole time
    with pytest.raises(EmptyGalleryError):
        service.recognize(photo(ANA_COLOR))

    enroll(repository, "Ana", ANA_COLOR)
    assert only_face(service.recognize(photo(ANA_COLOR))).person_name == "Ana"

    repository.delete_person(1)
    with pytest.raises(EmptyGalleryError):
        service.recognize(photo(ANA_COLOR))

    enroll(repository, "Bia", BIA_COLOR)
    assert only_face(service.recognize(photo(ANA_COLOR))).status is RecognitionStatus.UNKNOWN


def test_an_unknown_person_is_the_other_author_before_enrolling(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    result = only_face(recognizer(repository).recognize(photo(BIA_COLOR)))

    assert result.status is RecognitionStatus.UNKNOWN
    assert result.person_name is None


def test_an_unknown_result_does_not_reveal_the_closest_person(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    result = only_face(recognizer(repository, threshold=0.99).recognize(photo(QUERY_COLOR)))

    assert result.status is RecognitionStatus.UNKNOWN
    assert set(result.model_dump()) == {"status", "person_name", "similarity", "box"}
    assert "Ana" not in str(result.model_dump())
    assert 0 < result.similarity < 0.99


def test_the_log_has_the_face_count_then_the_status_and_similarity_of_each_face_but_no_name(
    repository: SqlitePersonRepository, caplog: pytest.LogCaptureFixture
) -> None:
    enroll(repository, "Ana Souza", ANA_COLOR)

    with caplog.at_level(logging.INFO, logger="app.core.recognition"):
        recognizer(repository).recognize(group_photo(ANA_COLOR, BIA_COLOR))

    image_record, first, second = caplog.records
    assert image_record.face_count == 2  # type: ignore[attr-defined]
    assert first.status == "recognized"  # type: ignore[attr-defined]
    assert first.similarity >= THRESHOLD  # type: ignore[attr-defined]
    assert second.status == "unknown"  # type: ignore[attr-defined]
    assert "Ana" not in caplog.text and "Souza" not in caplog.text


def test_decide_accepts_a_similarity_equal_to_the_threshold() -> None:
    result = decide(0.766, "Ana", 0.766)

    assert (result.status, result.person_name) == (RecognitionStatus.RECOGNIZED, "Ana")


def test_decide_rejects_a_similarity_just_below_the_threshold() -> None:
    result = decide(0.765, "Ana", 0.766)

    assert (result.status, result.person_name, result.similarity) == (
        RecognitionStatus.UNKNOWN,
        None,
        0.765,
    )


def test_decide_accepts_a_similarity_above_the_threshold() -> None:
    assert decide(0.99, "Bia", 0.766).person_name == "Bia"


# ---- several faces in one image (spec 003, A12 to A20; spec 004, A9 and A10) ----


def test_every_face_of_the_image_is_recognized_from_left_to_right(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    enroll(repository, "Bia", BIA_COLOR)

    response = recognizer(repository).recognize(group_photo(BIA_COLOR, ANA_COLOR))

    assert [face.person_name for face in response.faces] == ["Bia", "Ana"]
    assert [face.status for face in response.faces] == [RecognitionStatus.RECOGNIZED] * 2
    assert response.faces[0].box.left < response.faces[1].box.left


def test_a_known_and_an_unknown_face_in_the_same_image(repository: SqlitePersonRepository) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    known, stranger = recognizer(repository).recognize(group_photo(ANA_COLOR, BIA_COLOR)).faces

    assert (known.status, known.person_name) == (RecognitionStatus.RECOGNIZED, "Ana")
    assert (stranger.status, stranger.person_name) == (RecognitionStatus.UNKNOWN, None)
    assert "Ana" not in str(stranger.model_dump())


def test_two_unknown_faces_each_have_their_own_similarity_and_box(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    # Carla is 0.9 similar to Ana, so a stricter threshold leaves her and Bia both unknown.
    response = recognizer(repository, threshold=0.95).recognize(group_photo(BIA_COLOR, CARLA_COLOR))

    first, second = response.faces
    assert [face.status for face in response.faces] == [RecognitionStatus.UNKNOWN] * 2
    assert [face.person_name for face in response.faces] == [None, None]
    assert first.similarity != second.similarity
    assert first.box != second.box


def test_one_face_gives_a_list_of_one(repository: SqlitePersonRepository) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    response = recognizer(repository).recognize(photo(ANA_COLOR))

    assert len(response.faces) == 1


def test_the_same_person_in_two_faces_is_not_deduplicated(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    response = recognizer(repository).recognize(group_photo(ANA_COLOR, ANA_COLOR))

    assert [face.person_name for face in response.faces] == ["Ana", "Ana"]


def test_a_face_below_the_minimum_size_is_left_out(repository: SqlitePersonRepository) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    enroll(repository, "Bia", BIA_COLOR)

    response = recognizer(repository).recognize(group_photo(ANA_COLOR, (BIA_COLOR, 79)))

    assert [face.person_name for face in response.faces] == ["Ana"]


def test_only_the_biggest_faces_are_recognized_up_to_the_limit(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    response = recognizer(repository).recognize(group_photo(*[ANA_COLOR] * (MAX_FACES + 5)))

    assert len(response.faces) == MAX_FACES


def test_the_response_has_the_oriented_image_size_and_boxes_inside_it(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    response = recognizer(repository).recognize(group_photo(ANA_COLOR, BIA_COLOR))

    assert (response.image_width, response.image_height) == (204, 200)
    for face in response.faces:
        box = face.box
        assert 0 <= box.left < box.right <= response.image_width
        assert 0 <= box.top < box.bottom <= response.image_height


class CountingRepository(SqlitePersonRepository):
    """Counts how many times the gallery is read."""

    def __init__(self) -> None:
        super().__init__(":memory:")
        self.reads = 0

    def gallery(self, model_id: str):  # noqa: ANN201
        self.reads += 1
        return super().gallery(model_id)


def test_the_gallery_is_read_once_per_image_and_not_once_per_face() -> None:
    repository = CountingRepository()
    enroll(repository, "Ana", ANA_COLOR)
    repository.reads = 0

    recognizer(repository).recognize(group_photo(ANA_COLOR, BIA_COLOR, CARLA_COLOR))

    assert repository.reads == 1
    repository.close()
