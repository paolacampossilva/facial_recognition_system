"""Face detection with MTCNN behind a small protocol, so tests can inject fakes."""

from typing import Protocol, cast

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]
LANDMARK_COUNT: int = 5
# MTCNN.detect(..., landmarks=True) returns boxes, probabilities and landmarks, or Nones.
MTCNNOutput = tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]


class BoundingBox(BaseModel):
    """Face box in image pixels."""

    model_config = ConfigDict(frozen=True)

    left: float
    top: float
    right: float
    bottom: float

    @property
    def area(self) -> float:
        return max(self.right - self.left, 0.0) * max(self.bottom - self.top, 0.0)

    @property
    def center(self) -> Point:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)


class FaceDetection(BaseModel):
    """One detected face.

    Landmarks follow the MTCNN order: left eye, right eye, nose, left mouth
    corner and right mouth corner, in image pixels ("left" is the smaller x).
    """

    model_config = ConfigDict(frozen=True)

    box: BoundingBox
    probability: float
    landmarks: tuple[Point, ...] = Field(min_length=LANDMARK_COUNT, max_length=LANDMARK_COUNT)


class FaceDetector(Protocol):
    """Anything that finds every face in an RGB image."""

    def detect(self, image: Image.Image) -> list[FaceDetection]: ...


class MTCNNDetector:
    """facenet-pytorch's MTCNN returning every face with its landmarks."""

    def __init__(self, device: torch.device | str = "cpu", min_face_size: int = 20) -> None:
        self._mtcnn = MTCNN(min_face_size=min_face_size, keep_all=True, device=device)

    def detect(self, image: Image.Image) -> list[FaceDetection]:
        boxes, probabilities, points = cast(MTCNNOutput, self._mtcnn.detect(image, landmarks=True))
        if boxes is None or probabilities is None or points is None:
            return []

        detections = []
        for box, probability, face_points in zip(boxes, probabilities, points, strict=True):
            left, top, right, bottom = (float(value) for value in box)
            landmarks = tuple((float(x), float(y)) for x, y in face_points)
            detections.append(
                FaceDetection(
                    box=BoundingBox(left=left, top=top, right=right, bottom=bottom),
                    probability=float(probability),
                    landmarks=landmarks,
                )
            )
        return detections
