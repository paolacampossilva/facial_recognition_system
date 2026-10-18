"""Converts face crops into the tensors each backbone expects."""

from enum import StrEnum

import numpy as np
import torch
from PIL import Image


class Normalization(StrEnum):
    FACENET = "facenet"
    IMAGENET = "imagenet"


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def to_tensor(face: Image.Image, normalization: Normalization) -> torch.Tensor:
    """Turns an RGB face crop into a normalized ``(3, H, W)`` float tensor.

    Args:
        face: RGB face crop.
        normalization: ``facenet`` applies facenet-pytorch's fixed standardization
            ``(x - 127.5) / 128``; ``imagenet`` scales to [0, 1] and applies the
            ImageNet channel mean and standard deviation (ResNet-50).

    Returns:
        The normalized tensor.
    """
    pixels = torch.from_numpy(np.asarray(face.convert("RGB"), dtype=np.float32))
    tensor = pixels.permute(2, 0, 1).contiguous()

    if normalization is Normalization.FACENET:
        return (tensor - 127.5) / 128.0

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor / 255.0 - mean) / std
