"""Backbones A and B of decision D4, behind one interface that returns unit-norm embeddings.

Both wrappers take a batch of 160x160 face crops already normalized with
``embedder.preprocess`` and return one L2-normalized embedding per face, so the
evaluation only compares vectors and never needs to know which backbone made them.
"""

from enum import StrEnum

import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch import nn
from torch.nn.functional import interpolate, normalize
from torchvision.models import ResNet50_Weights, resnet50

from facial_recognition_system.preprocessing import Normalization, to_tensor

FACE_SIZE: int = 160
RESNET_INPUT_SIZE: int = 224


class Backbone(StrEnum):
    FACENET = "facenet"
    RESNET50 = "resnet50"


class FaceEmbedder(nn.Module):
    """Maps normalized face crops ``(N, 3, H, W)`` to L2-normalized embeddings ``(N, D)``."""

    model_id: str
    embedding_dim: int
    normalization: Normalization

    def preprocess(self, face: Image.Image) -> torch.Tensor:
        """Turns one RGB face crop into the ``(3, H, W)`` tensor this backbone expects."""
        return to_tensor(face, self.normalization)

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        if faces.ndim != 4 or faces.shape[1] != 3:
            raise ValueError(f"expected a (N, 3, H, W) batch, got {tuple(faces.shape)}")
        return normalize(self.extract(faces), dim=1)

    def extract(self, faces: torch.Tensor) -> torch.Tensor:
        """Raw, unnormalized embeddings of the batch."""
        raise NotImplementedError


class FaceNetEmbedder(FaceEmbedder):
    """Backbone A: facenet-pytorch InceptionResnetV1 pretrained on VGGFace2, 512 dimensions.

    The network is exposed as ``self.network`` so the partial fine-tuning of
    decision D5 can unfreeze ``block8``, ``last_linear`` and ``last_bn``.
    """

    embedding_dim = 512
    normalization = Normalization.FACENET

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.model_id = "facenet-vggface2" if pretrained else "facenet-random"
        self.network = InceptionResnetV1(pretrained="vggface2" if pretrained else None)

    def extract(self, faces: torch.Tensor) -> torch.Tensor:
        return self.network(faces)


class ResNet50Embedder(FaceEmbedder):
    """Backbone B: torchvision ResNet-50 with ImageNet weights and no ``fc``, 2048 dimensions.

    The embedding is the global average pooling output, with no projection, so
    the comparison measures only what the ImageNet pretraining learned. Crops are
    resized to 224x224 inside the model, which lets both backbones share the
    same 160x160 crops (bilinear resizing commutes with the per-channel
    normalization, so normalizing first gives the same input).
    """

    embedding_dim = 2048
    normalization = Normalization.IMAGENET

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.model_id = "resnet50-imagenet1k-v2" if pretrained else "resnet50-random"
        self.network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.network.fc = nn.Identity()

    def extract(self, faces: torch.Tensor) -> torch.Tensor:
        if faces.shape[-2:] != (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE):
            faces = interpolate(
                faces,
                size=(RESNET_INPUT_SIZE, RESNET_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        return self.network(faces)


def build_embedder(backbone: Backbone | str, pretrained: bool = True) -> FaceEmbedder:
    """Creates the wrapper of a backbone, in eval mode.

    Args:
        backbone: ``facenet`` (A) or ``resnet50`` (B).
        pretrained: Loads the pretrained weights; ``False`` gives random weights for tests.

    Returns:
        The embedder, in eval mode and on the CPU.
    """
    match Backbone(backbone):
        case Backbone.FACENET:
            embedder: FaceEmbedder = FaceNetEmbedder(pretrained)
        case Backbone.RESNET50:
            embedder = ResNet50Embedder(pretrained)
    return embedder.eval()
