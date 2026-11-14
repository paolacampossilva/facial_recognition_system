"""Integration tests of the services with the real MTCNN and the real FaceNet, on public LFW photos.

Only LFW photos (public data) are used, never the private dataset. The model is the E06 checkpoint
when it is in this machine, else the pretrained FaceNet when its weights are cached; with neither,
the tests are skipped, as they are when the LFW photos are not downloaded.
"""

import os
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.core.deletion import DeletionService
from app.core.encoder import FaceEncoder
from app.core.enrollment import EnrollmentService
from app.core.errors import DuplicatePersonError, EmptyGalleryError
from app.core.models import RecognitionStatus
from app.core.recognition import RecognitionService
from app.embedder import TorchFaceEmbedder
from app.storage.repository import SqlitePersonRepository
from facial_recognition_system.preprocessing import (
    FacePreprocessor,
    InvalidImageError,
    MTCNNDetector,
    MultipleFacesError,
    NoFaceDetectedError,
    PreprocessingConfig,
)
from tests.app_support import request

pytestmark = pytest.mark.integration

LFW_DIR: Path = Path("data/public/lfw")
E06_CHECKPOINT: Path = Path("checkpoints/E06/best.pt")
TORCH_HOME: Path = Path(os.getenv("TORCH_HOME", Path.home() / ".cache" / "torch"))
PRETRAINED_WEIGHTS: Path = TORCH_HOME / "checkpoints" / "20180402-114759-vggface2.pt"

# The 0.766 of the application was chosen on the private set; these tests use LFW photos, so they
# use a threshold between the same-person and different-person similarities seen on LFW (D08).
LFW_THRESHOLD: float = 0.5

# Identities of the LFW test split (data/splits/lfw_test.txt), which the E06 never trained on.
RUMSFELD: str = "Donald_Rumsfeld"
CHIRAC: str = "Jacques_Chirac"
ANNAN: str = "Kofi_Annan"


def lfw_photo(identity: str, number: int) -> bytes:
    """The bytes of one original LFW photo (`<identity>_<number:04d>.jpg`)."""
    path = LFW_DIR / identity / f"{identity}_{number:04d}.jpg"
    if not path.is_file():
        pytest.skip("LFW not downloaded; run scripts/download_datasets.py")
    return path.read_bytes()


def blank_photo() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (250, 250), (128, 128, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def two_faces_photo() -> bytes:
    """Two different LFW photos side by side, so the detector finds two faces."""
    left = Image.open(BytesIO(lfw_photo(RUMSFELD, 1))).convert("RGB")
    right = Image.open(BytesIO(lfw_photo(CHIRAC, 1))).convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def encoder() -> FaceEncoder:
    """Real detector and real embedder, loaded once for the whole module."""
    if E06_CHECKPOINT.is_file():
        embedder = TorchFaceEmbedder(E06_CHECKPOINT)
    elif PRETRAINED_WEIGHTS.is_file():
        embedder = TorchFaceEmbedder()
    else:
        pytest.skip("neither checkpoints/E06/best.pt nor the pretrained FaceNet weights are here")
    preprocessor = FacePreprocessor(MTCNNDetector(device="cpu"), PreprocessingConfig())
    return FaceEncoder(preprocessor, embedder)


@pytest.fixture
def enrollment(encoder: FaceEncoder, repository: SqlitePersonRepository) -> EnrollmentService:
    return EnrollmentService(encoder, repository, LFW_THRESHOLD)


@pytest.fixture
def recognition(encoder: FaceEncoder, repository: SqlitePersonRepository) -> RecognitionService:
    return RecognitionService(encoder, repository, LFW_THRESHOLD)


@pytest.fixture
def deletion(repository: SqlitePersonRepository) -> DeletionService:
    return DeletionService(repository)


def enroll_rumsfeld_and_chirac(enrollment: EnrollmentService) -> tuple[int, int]:
    rumsfeld = enrollment.enroll(
        request("Rumsfeld", *(lfw_photo(RUMSFELD, number) for number in (1, 2, 3)))
    )
    chirac = enrollment.enroll(request("Chirac", *(lfw_photo(CHIRAC, number) for number in (1, 2))))
    return rumsfeld.person_id, chirac.person_id


def test_each_enrolled_person_is_recognized_by_a_new_photo(
    enrollment: EnrollmentService, recognition: RecognitionService
) -> None:
    enroll_rumsfeld_and_chirac(enrollment)

    first = recognition.recognize(lfw_photo(RUMSFELD, 4))
    second = recognition.recognize(lfw_photo(CHIRAC, 3))

    assert (first.status, first.person_name) == (RecognitionStatus.RECOGNIZED, "Rumsfeld")
    assert (second.status, second.person_name) == (RecognitionStatus.RECOGNIZED, "Chirac")
    assert first.similarity >= LFW_THRESHOLD
    assert second.similarity >= LFW_THRESHOLD


def test_a_person_who_is_not_enrolled_is_unknown_and_has_no_name(
    enrollment: EnrollmentService, recognition: RecognitionService
) -> None:
    enroll_rumsfeld_and_chirac(enrollment)

    result = recognition.recognize(lfw_photo(ANNAN, 1))

    assert result.status == RecognitionStatus.UNKNOWN
    assert result.person_name is None
    assert result.similarity < LFW_THRESHOLD


def test_enrolling_a_face_that_is_already_enrolled_under_another_name_is_refused(
    enrollment: EnrollmentService, repository: SqlitePersonRepository
) -> None:
    enroll_rumsfeld_and_chirac(enrollment)

    with pytest.raises(DuplicatePersonError) as refused:
        enrollment.enroll(request("Another name", lfw_photo(RUMSFELD, 4)))

    assert refused.value.reason == "similarity"
    assert refused.value.name == "Rumsfeld"
    assert [person.name for person in repository.list_persons()] == ["Rumsfeld", "Chirac"]


@pytest.mark.parametrize(
    ("bad_photo", "error"),
    [
        pytest.param(blank_photo, NoFaceDetectedError, id="no-face"),
        pytest.param(two_faces_photo, MultipleFacesError, id="two-faces"),
        pytest.param(lambda: b"not an image", InvalidImageError, id="not-an-image"),
    ],
)
def test_a_bad_photo_fails_enrollment_and_recognition_with_its_own_error(
    bad_photo: Callable[[], bytes],
    error: type[Exception],
    enrollment: EnrollmentService,
    recognition: RecognitionService,
    repository: SqlitePersonRepository,
) -> None:
    enrollment.enroll(request("Rumsfeld", lfw_photo(RUMSFELD, 1)))

    with pytest.raises(error):
        recognition.recognize(bad_photo())
    with pytest.raises(error):
        enrollment.enroll(request("Chirac", lfw_photo(CHIRAC, 1), bad_photo()))

    # All or nothing: the good photo of the failed enrollment was not stored either.
    assert [person.name for person in repository.list_persons()] == ["Rumsfeld"]


def test_recognition_with_nobody_enrolled_is_the_empty_gallery_error(
    recognition: RecognitionService,
) -> None:
    with pytest.raises(EmptyGalleryError):
        recognition.recognize(lfw_photo(RUMSFELD, 1))


def test_after_deleting_a_person_their_photo_is_unknown_not_someone_else(
    enrollment: EnrollmentService,
    recognition: RecognitionService,
    deletion: DeletionService,
) -> None:
    rumsfeld_id, _ = enroll_rumsfeld_and_chirac(enrollment)

    deleted = deletion.delete(rumsfeld_id)
    result = recognition.recognize(lfw_photo(RUMSFELD, 4))

    assert deleted.deleted_embeddings == 3
    assert result.status == RecognitionStatus.UNKNOWN
    assert result.person_name is None
