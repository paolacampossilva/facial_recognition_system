from pathlib import Path

import pytest
import torch
from PIL import Image

from facial_recognition_system.models import (
    FACE_SIZE,
    Backbone,
    FaceEmbedder,
    FaceNetEmbedder,
    ResNet50Embedder,
    build_embedder,
)
from facial_recognition_system.preprocessing import Normalization, load_image

LFW_CROPS: Path = Path("data/public/lfw_crop160")
SAME_PERSON: tuple[str, str] = (
    "George_W_Bush/George_W_Bush_0001.png",
    "George_W_Bush/George_W_Bush_0002.png",
)
OTHER_PERSON: str = "Aaron_Eckhart/Aaron_Eckhart_0001.png"

EXPECTED_DIMS: dict[Backbone, int] = {Backbone.FACENET: 512, Backbone.RESNET50: 2048}


@pytest.fixture(scope="module", params=list(Backbone))
def backbone(request: pytest.FixtureRequest) -> Backbone:
    return request.param


@pytest.fixture(scope="module")
def random_embedder(backbone: Backbone) -> FaceEmbedder:
    return build_embedder(backbone, pretrained=False)


def embed(embedder: FaceEmbedder, faces: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return embedder(faces)


@pytest.mark.unit
def test_embedding_has_backbone_dimension(
    backbone: Backbone, random_embedder: FaceEmbedder
) -> None:
    embeddings = embed(random_embedder, torch.randn(3, 3, FACE_SIZE, FACE_SIZE))

    expected = EXPECTED_DIMS[backbone]
    assert random_embedder.embedding_dim == expected
    assert embeddings.shape == (3, expected)
    assert embeddings.dtype == torch.float32


@pytest.mark.unit
def test_embeddings_are_l2_normalized(random_embedder: FaceEmbedder) -> None:
    embeddings = embed(random_embedder, torch.randn(3, 3, FACE_SIZE, FACE_SIZE))

    assert torch.allclose(embeddings.norm(dim=1), torch.ones(3), atol=1e-5)


@pytest.mark.unit
def test_preprocess_gives_backbone_input(random_embedder: FaceEmbedder) -> None:
    face = Image.new("RGB", (FACE_SIZE, FACE_SIZE), (200, 100, 50))

    tensor = random_embedder.preprocess(face)

    assert tensor.shape == (3, FACE_SIZE, FACE_SIZE)
    assert embed(random_embedder, tensor.unsqueeze(0)).shape[0] == 1


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(3, FACE_SIZE, FACE_SIZE), (2, 1, FACE_SIZE, FACE_SIZE)])
def test_rejects_batch_that_is_not_rgb_nchw(
    random_embedder: FaceEmbedder, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="expected a"):
        random_embedder(torch.randn(*shape))


@pytest.mark.unit
def test_build_embedder_returns_eval_mode_wrapper() -> None:
    facenet = build_embedder("facenet", pretrained=False)
    resnet = build_embedder(Backbone.RESNET50, pretrained=False)

    assert isinstance(facenet, FaceNetEmbedder)
    assert isinstance(resnet, ResNet50Embedder)
    assert not facenet.training
    assert not resnet.training
    assert facenet.normalization is Normalization.FACENET
    assert resnet.normalization is Normalization.IMAGENET


@pytest.mark.unit
def test_build_embedder_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError):
        build_embedder("vgg16", pretrained=False)


@pytest.mark.unit
def test_resnet_gives_same_embedding_for_160_and_224_inputs() -> None:
    resnet = build_embedder(Backbone.RESNET50, pretrained=False)
    faces = torch.randn(2, 3, FACE_SIZE, FACE_SIZE)
    resized = torch.nn.functional.interpolate(
        faces, size=(224, 224), mode="bilinear", align_corners=False
    )

    assert torch.allclose(embed(resnet, faces), embed(resnet, resized), atol=1e-5)


@pytest.mark.integration
def test_pretrained_embedder_on_lfw_crops(backbone: Backbone) -> None:
    if not (LFW_CROPS / OTHER_PERSON).exists():
        pytest.skip("LFW crops missing; run scripts/preprocess_faces.py")
    embedder = build_embedder(backbone)
    paths = [*SAME_PERSON, OTHER_PERSON]
    faces = torch.stack([embedder.preprocess(load_image(LFW_CROPS / path)) for path in paths])

    embeddings = embed(embedder, faces)

    assert embeddings.shape == (3, EXPECTED_DIMS[backbone])
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(3), atol=1e-5)
    if backbone is Backbone.FACENET:
        same = embeddings[0] @ embeddings[1]
        different = embeddings[0] @ embeddings[2]
        assert same > different, "the face-pretrained FaceNet must separate two people"
