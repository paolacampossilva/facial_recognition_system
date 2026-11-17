"""Tests of the real embedder: unit tests on a random checkpoint, integration on real weights."""

import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app.core.encoder import FaceEncoder
from app.core.enrollment import EnrollmentService
from app.core.errors import ModelLoadError
from app.core.models import RecognitionStatus
from app.core.recognition import RecognitionService
from app.embedder import PRETRAINED_MODEL_ID, TorchFaceEmbedder, pick_device
from app.storage.repository import MEMORY_DATABASE, SqlitePersonRepository
from facial_recognition_system.models import FACE_SIZE, build_embedder
from facial_recognition_system.preprocessing import FacePreprocessor, PreprocessingConfig
from facial_recognition_system.training import load_checkpoint, save_checkpoint
from tests.app_support import ANA_COLOR, BIA_COLOR, THRESHOLD, ScriptedDetector, photo, request

DIMENSION: int = 512
LFW_CROPS: Path = Path("data/public/lfw_crop160")
SAME_PERSON: tuple[str, str] = (
    "George_W_Bush/George_W_Bush_0001.png",
    "George_W_Bush/George_W_Bush_0002.png",
)
OTHER_PERSON: str = "Aaron_Eckhart/Aaron_Eckhart_0001.png"
E06_CHECKPOINT: Path = Path("checkpoints/E06/best.pt")
TORCH_HOME: Path = Path(os.getenv("TORCH_HOME", Path.home() / ".cache" / "torch"))
PRETRAINED_WEIGHTS: Path = TORCH_HOME / "checkpoints" / "20180402-114759-vggface2.pt"


def face(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (FACE_SIZE, FACE_SIZE), color)


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(first @ second)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random FaceNet in the real checkpoint format, in a folder named like an experiment."""
    torch.manual_seed(0)
    network = build_embedder("facenet", pretrained=False)
    path = tmp_path_factory.mktemp("checkpoints") / "E06" / "best.pt"
    save_checkpoint(path, network, nn.Identity(), epoch=1, val_eer=0.01)
    return path


@pytest.fixture(scope="module")
def embedder(checkpoint: Path) -> TorchFaceEmbedder:
    return TorchFaceEmbedder(checkpoint)


@pytest.mark.unit
def test_an_embedding_is_a_float32_vector_of_512_values_with_unit_norm(
    embedder: TorchFaceEmbedder,
) -> None:
    vector = embedder.embed(face(ANA_COLOR))

    assert vector.shape == (DIMENSION,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.unit
def test_the_same_face_gives_the_same_embedding_and_other_faces_do_not(
    embedder: TorchFaceEmbedder,
) -> None:
    first = embedder.embed(face(ANA_COLOR))

    assert np.array_equal(first, embedder.embed(face(ANA_COLOR)))
    assert not np.array_equal(first, embedder.embed(face(BIA_COLOR)))


@pytest.mark.unit
def test_the_weights_of_the_checkpoint_are_the_ones_in_use(
    checkpoint: Path, embedder: TorchFaceEmbedder
) -> None:
    by_hand = build_embedder("facenet", pretrained=False)
    load_checkpoint(checkpoint, by_hand)
    random_network = build_embedder("facenet", pretrained=False)  # a different random init
    tensor = by_hand.preprocess(face(ANA_COLOR)).unsqueeze(0)

    with torch.no_grad():
        expected = by_hand(tensor)[0].numpy()
        unrelated = random_network(tensor)[0].numpy()

    ours = embedder.embed(face(ANA_COLOR))
    assert np.allclose(ours, expected, atol=1e-6)
    assert not np.allclose(ours, unrelated, atol=1e-3)


@pytest.mark.unit
def test_the_model_id_comes_from_the_experiment_folder_of_the_checkpoint(
    embedder: TorchFaceEmbedder,
) -> None:
    assert embedder.model_id == "facenet-vggface2-E06"
    assert embedder.model_id != PRETRAINED_MODEL_ID  # E06 and E01 vectors must never mix


@pytest.mark.unit
def test_the_model_id_can_be_overridden(checkpoint: Path) -> None:
    assert TorchFaceEmbedder(checkpoint, model_id="my-model").model_id == "my-model"


@pytest.mark.unit
def test_it_works_inside_the_encoder_and_both_services(
    embedder: TorchFaceEmbedder,
) -> None:
    repository = SqlitePersonRepository(MEMORY_DATABASE)
    encoder = FaceEncoder(FacePreprocessor(ScriptedDetector(), PreprocessingConfig()), embedder)
    EnrollmentService(encoder, repository, THRESHOLD).enroll(request("Ana", photo(ANA_COLOR)))

    result = RecognitionService(encoder, repository, THRESHOLD).recognize(photo(ANA_COLOR))

    assert repository.gallery("facenet-vggface2-E06").labels == ["Ana"]
    assert repository.gallery(PRETRAINED_MODEL_ID).labels == []
    assert result.status is RecognitionStatus.RECOGNIZED
    assert result.person_name == "Ana"
    repository.close()


@pytest.mark.unit
def test_several_threads_get_the_same_embedding(embedder: TorchFaceEmbedder) -> None:
    expected = embedder.embed(face(ANA_COLOR))
    start = threading.Barrier(4)

    def work() -> np.ndarray:
        start.wait()
        return embedder.embed(face(ANA_COLOR))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [future.result() for future in [pool.submit(work) for _ in range(4)]]

    assert all(np.array_equal(result, expected) for result in results)


@pytest.mark.unit
def test_a_missing_checkpoint_names_the_path_and_how_to_train_it(tmp_path: Path) -> None:
    missing = tmp_path / "E06" / "best.pt"

    with pytest.raises(ModelLoadError, match="checkpoint not found") as error:
        TorchFaceEmbedder(missing)

    assert str(missing) in str(error.value)
    assert "scripts/train.py" in str(error.value)


@pytest.mark.unit
def test_a_file_that_is_not_a_checkpoint_is_a_load_error(tmp_path: Path) -> None:
    broken = tmp_path / "best.pt"
    broken.write_bytes(b"this is not a checkpoint")

    with pytest.raises(ModelLoadError, match="not a valid FaceNet checkpoint"):
        TorchFaceEmbedder(broken)


@pytest.mark.unit
def test_a_checkpoint_of_another_network_is_a_load_error(tmp_path: Path) -> None:
    resnet = build_embedder("resnet50", pretrained=False)
    path = tmp_path / "R50" / "best.pt"
    save_checkpoint(path, resnet, nn.Identity(), epoch=1, val_eer=0.5)

    with pytest.raises(ModelLoadError, match="not a valid FaceNet checkpoint"):
        TorchFaceEmbedder(path)


@pytest.mark.unit
def test_a_checkpoint_trained_from_the_pretrained_facenet_loads_without_its_classifier(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    network = build_embedder("facenet", pretrained=False)
    weights = cast(nn.Module, network.network).state_dict()
    weights["logits.weight"] = torch.zeros(8631, 512)
    weights["logits.bias"] = torch.zeros(8631)
    path = tmp_path / "E06" / "best.pt"
    path.parent.mkdir()
    torch.save({"epoch": 1, "network": weights}, path)

    loaded = TorchFaceEmbedder(path)

    assert loaded.embed(face(ANA_COLOR)).shape == (DIMENSION,)


@pytest.mark.unit
def test_a_checkpoint_missing_a_layer_of_the_network_is_still_a_load_error(
    tmp_path: Path,
) -> None:
    network = build_embedder("facenet", pretrained=False)
    weights = cast(nn.Module, network.network).state_dict()
    del weights["last_linear.weight"]
    weights["logits.weight"] = torch.zeros(8631, 512)
    path = tmp_path / "E06" / "best.pt"
    path.parent.mkdir()
    torch.save({"epoch": 1, "network": weights}, path)

    with pytest.raises(ModelLoadError, match="not a valid FaceNet checkpoint"):
        TorchFaceEmbedder(path)


@pytest.mark.unit
def test_a_checkpoint_without_a_network_is_a_load_error(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    torch.save({"epoch": 1}, path)

    with pytest.raises(ModelLoadError):
        TorchFaceEmbedder(path)


@pytest.mark.unit
@pytest.mark.parametrize(("preference", "expected"), [("cpu", "cpu"), ("auto", "cpu")])
def test_the_device_without_a_gpu_is_the_cpu(
    monkeypatch: pytest.MonkeyPatch, preference: str, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert pick_device(preference) == expected


@pytest.mark.unit
def test_auto_picks_the_gpu_when_there_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert pick_device("auto") == "cuda"
    assert pick_device("cuda") == "cuda"


@pytest.mark.unit
def test_asking_for_cuda_without_a_gpu_is_a_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ModelLoadError, match="no GPU"):
        pick_device("cuda")


@pytest.mark.unit
def test_an_unknown_device_is_a_load_error() -> None:
    with pytest.raises(ModelLoadError, match="unknown device"):
        pick_device("tpu")


@pytest.fixture(scope="module")
def lfw_faces() -> Iterator[dict[str, Image.Image]]:
    if not (LFW_CROPS / SAME_PERSON[0]).is_file():
        pytest.skip("the LFW crops are not in data/public/lfw_crop160")
    names = [*SAME_PERSON, OTHER_PERSON]
    yield {name: Image.open(LFW_CROPS / name).convert("RGB") for name in names}


def check_same_person_is_closer(embedder: TorchFaceEmbedder, faces: dict[str, Image.Image]) -> None:
    first, second = (embedder.embed(faces[name]) for name in SAME_PERSON)
    other = embedder.embed(faces[OTHER_PERSON])

    assert first.shape == (DIMENSION,)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    assert cosine(first, second) > cosine(first, other)


@pytest.mark.integration
@pytest.mark.skipif(not E06_CHECKPOINT.is_file(), reason="checkpoints/E06/best.pt is not here")
def test_the_real_e06_checkpoint_puts_the_same_person_closer(
    lfw_faces: dict[str, Image.Image],
) -> None:
    embedder = TorchFaceEmbedder(E06_CHECKPOINT)

    assert embedder.model_id == "facenet-vggface2-E06"
    check_same_person_is_closer(embedder, lfw_faces)


@pytest.mark.integration
@pytest.mark.skipif(
    not PRETRAINED_WEIGHTS.is_file(), reason="the pretrained FaceNet weights are not in the cache"
)
def test_the_pretrained_facenet_puts_the_same_person_closer(
    lfw_faces: dict[str, Image.Image],
) -> None:
    embedder = TorchFaceEmbedder()

    assert embedder.model_id == PRETRAINED_MODEL_ID
    check_same_person_is_closer(embedder, lfw_faces)
