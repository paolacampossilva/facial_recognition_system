"""Verifica se o ambiente consegue executar o pipeline de detecção e embedding.

Rodar em cada máquina nova:
    pytest                  # testes em CPU
    pytest -m gpu -s        # exige GPU; -s mostra qual GPU foi usada
"""

import pytest
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

EMBEDDING_DIM = 512
FACE_SIZE = 160
NO_CUDA_MSG = "CUDA indisponível: verifique se o torch foi instalado com cu128"


@pytest.fixture(scope="module")
def blank_image():
    return Image.new("RGB", (320, 240), color=(128, 128, 128))


def _check_embedding(device: torch.device) -> None:
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    batch = torch.randn(2, 3, FACE_SIZE, FACE_SIZE, device=device)

    with torch.no_grad():
        embeddings = model(batch)

    assert embeddings.shape == (2, EMBEDDING_DIM)
    norms = embeddings.norm(dim=1).cpu()
    assert torch.allclose(norms, torch.ones(2), atol=1e-4), "embeddings devem ser L2-normalizados"


def _check_detector(device: torch.device, image: Image.Image) -> None:
    mtcnn = MTCNN(image_size=FACE_SIZE, device=device)
    assert mtcnn(image) is None, "imagem sem rosto não deve gerar detecção"


@pytest.mark.integration
def test_embedding_cpu():
    _check_embedding(torch.device("cpu"))


@pytest.mark.integration
def test_detector_cpu(blank_image):
    _check_detector(torch.device("cpu"), blank_image)


@pytest.mark.integration
@pytest.mark.gpu
def test_embedding_gpu():
    # is_available() pode ser True mesmo quando o torch não tem kernels para a
    # arquitetura da placa; por isso o modelo é executado.
    assert torch.cuda.is_available(), NO_CUDA_MSG
    print(
        f"\nGPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}"
        f" | CUDA {torch.version.cuda}"
    )
    _check_embedding(torch.device("cuda"))


@pytest.mark.integration
@pytest.mark.gpu
def test_detector_gpu(blank_image):
    assert torch.cuda.is_available(), NO_CUDA_MSG
    _check_detector(torch.device("cuda"), blank_image)
