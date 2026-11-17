"""Builds the real services the API calls, from the configuration (spec 007, issue #49)."""

import logging
from dataclasses import dataclass

from app.config import Settings
from app.core.deletion import DeletionService
from app.core.encoder import FaceEncoder
from app.core.enrollment import EnrollmentService
from app.core.recognition import RecognitionService
from app.embedder import TorchFaceEmbedder, pick_device
from app.storage.repository import SqlitePersonRepository
from facial_recognition_system.preprocessing import (
    FacePreprocessor,
    MTCNNDetector,
    PreprocessingConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealServices:
    """The four services of the `Services` protocol, over one repository."""

    enrollment: EnrollmentService
    recognition: RecognitionService
    deletion: DeletionService
    persons: SqlitePersonRepository


def load_encoder(settings: Settings) -> FaceEncoder:
    """Loads the face detector and the configured embedder, once, at startup.

    Raises:
        ModelLoadError: The checkpoint is missing or invalid, or the device is unavailable.
    """
    device = pick_device(settings.device)
    embedder = TorchFaceEmbedder(settings.checkpoint_path, device=device)
    preprocessor = FacePreprocessor(MTCNNDetector(device=device), PreprocessingConfig())
    logger.info("model loaded", extra={"model_id": embedder.model_id, "device": device})
    return FaceEncoder(preprocessor, embedder)


def build_services(
    settings: Settings, encoder: FaceEncoder, repository: SqlitePersonRepository
) -> RealServices:
    """Wires the services to the encoder and the repository, with the configured threshold."""
    return RealServices(
        enrollment=EnrollmentService(encoder, repository, settings.threshold),
        recognition=RecognitionService(
            encoder, repository, settings.threshold, settings.min_face_px
        ),
        deletion=DeletionService(repository),
        persons=repository,
    )
