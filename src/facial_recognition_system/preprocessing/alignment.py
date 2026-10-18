"""Turns a detected face into a square crop.

Two modes are available:

- ``crop``: the facenet-pytorch crop (box plus margin, resized to a square),
  which is the input distribution the pretrained FaceNet was trained on.
- ``similarity``: rotates, scales and translates the face so the five landmarks
  land on the ArcFace reference template, removing head roll.
"""

from enum import StrEnum

import numpy as np
from PIL import Image

from facial_recognition_system.preprocessing.detection import BoundingBox, Point


class AlignmentMode(StrEnum):
    CROP = "crop"
    SIMILARITY = "similarity"


TEMPLATE_SIZE: int = 112
# Standard ArcFace landmark positions for a 112x112 crop, in MTCNN landmark order.
ARCFACE_TEMPLATE: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def crop_face(image: Image.Image, box: BoundingBox, image_size: int, margin: int) -> Image.Image:
    """Crops the box plus margin and resizes it to a square, as facenet-pytorch does.

    Args:
        image: RGB source image.
        box: Detected face box.
        image_size: Side of the output square in pixels.
        margin: Margin around the box, measured in pixels of the output image.

    Returns:
        The ``image_size`` x ``image_size`` face crop.
    """
    margin_x = margin * (box.right - box.left) / (image_size - margin)
    margin_y = margin * (box.bottom - box.top) / (image_size - margin)
    width, height = image.size

    crop_box = (
        int(max(box.left - margin_x / 2, 0)),
        int(max(box.top - margin_y / 2, 0)),
        int(min(box.right + margin_x / 2, width)),
        int(min(box.bottom + margin_y / 2, height)),
    )
    return image.crop(crop_box).resize((image_size, image_size), Image.Resampling.BILINEAR)


def estimate_similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Finds the rotation, uniform scale and translation that best map source onto target.

    Uses the Umeyama least-squares solution without reflection.

    Args:
        source: ``(n, 2)`` points in the input image.
        target: ``(n, 2)`` corresponding points in the output image.

    Returns:
        ``(2, 3)`` affine matrix ``[s*R | t]`` such that ``target ≈ s*R @ source + t``.
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection_fix = np.diag([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ reflection_fix @ vt

    source_variance = (source_centered**2).sum() / len(source)
    scale = np.trace(np.diag(singular_values) @ reflection_fix) / source_variance
    translation = target_mean - scale * rotation @ source_mean

    return np.hstack([scale * rotation, translation[:, None]])


def align_face(image: Image.Image, landmarks: tuple[Point, ...], image_size: int) -> Image.Image:
    """Warps the face so its landmarks match the ArcFace template scaled to ``image_size``.

    Args:
        image: RGB source image.
        landmarks: The five MTCNN landmarks in image pixels.
        image_size: Side of the output square in pixels.

    Returns:
        The aligned ``image_size`` x ``image_size`` face.
    """
    template = ARCFACE_TEMPLATE * image_size / TEMPLATE_SIZE
    forward = estimate_similarity_transform(np.asarray(landmarks, dtype=np.float64), template)

    # PIL maps each output pixel back to the input, so it needs the inverse transform.
    linear_inverse = np.linalg.inv(forward[:, :2])
    translation_inverse = -linear_inverse @ forward[:, 2]
    inverse = np.hstack([linear_inverse, translation_inverse[:, None]])

    return image.transform(
        (image_size, image_size),
        Image.Transform.AFFINE,
        data=tuple(inverse.ravel()),
        resample=Image.Resampling.BILINEAR,
    )
