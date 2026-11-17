"""The real face embedder behind the application's `Embedder` protocol (decision D16).

Loading the model is slow and can fail (no checkpoint, no GPU), so it happens once, when the
application starts, and failures surface as `ModelLoadError` for the edge to report.
"""

import threading
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch import nn

from app.core.errors import ModelLoadError
from facial_recognition_system.models import FaceEmbedder, build_embedder

# The FaceNet pretrained on VGGFace2, with no fine-tuning (experiment E01).
PRETRAINED_MODEL_ID: str = "facenet-vggface2"
TRAIN_HINT: str = "train it with: python scripts/train.py --config configs/E06.yaml"
CLASSIFIER_PREFIX: str = "logits."
DEVICE_PREFERENCES: frozenset[str] = frozenset({"auto", "cpu", "cuda"})


def pick_device(preference: str) -> str:
    """Resolves a device preference to the device the model runs on.

    Args:
        preference: `auto` (the GPU when there is one), `cpu` or `cuda`.

    Raises:
        ModelLoadError: The preference is unknown, or `cuda` was asked for without a GPU.
    """
    if preference not in DEVICE_PREFERENCES:
        raise ModelLoadError(f"unknown device {preference!r}; use auto, cpu or cuda")
    if preference == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("cuda was requested but no GPU is available; use auto or cpu")
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


class TorchFaceEmbedder:
    """FaceNet embeddings, from the fine-tuned checkpoint of the chosen model or the pretrained one.

    The `model_id` tells embeddings of different weights apart: the fine-tuned E06 and the
    pretrained E01 are both a FaceNet, but their vectors are not in the same space, so mixing them
    in one gallery would make recognition wrong without any error (decision D12). The id of a
    checkpoint therefore includes the experiment folder it lives in (`facenet-vggface2-E06`).
    """

    def __init__(
        self,
        checkpoint: Path | None = None,
        device: str = "cpu",
        model_id: str | None = None,
    ) -> None:
        """Loads the model once.

        Args:
            checkpoint: The `best.pt` of the chosen experiment (`checkpoints/E06/best.pt`). None
                uses the pretrained FaceNet, which downloads its weights the first time.
            device: `auto`, `cpu` or `cuda`.
            model_id: Overrides the id stored with each embedding. By default it comes from the
                checkpoint folder, or is the pretrained model's id when there is no checkpoint.

        Raises:
            ModelLoadError: The checkpoint is missing or unreadable, or the device is unavailable.
        """
        self._device = pick_device(device)
        if checkpoint is None:
            embedder = build_embedder("facenet", pretrained=True)
            default_id = PRETRAINED_MODEL_ID
        else:
            # The checkpoint holds the whole network, so the pretrained weights are not needed.
            embedder = build_embedder("facenet", pretrained=False)
            self._load_checkpoint(checkpoint, embedder)
            default_id = f"{PRETRAINED_MODEL_ID}-{checkpoint.resolve().parent.name}"
        self.model_id: str = model_id or default_id
        self._embedder: FaceEmbedder = embedder.to(self._device).eval()
        self._lock = threading.Lock()

    def embed(self, face: Image.Image) -> np.ndarray:
        """Returns the embedding of a face crop: a float32 vector of 512 values with unit norm."""
        tensor = self._embedder.preprocess(face).unsqueeze(0).to(self._device)
        # One pass at a time: the API calls this from worker threads and a GPU gains nothing more.
        with self._lock, torch.no_grad():
            output = self._embedder(tensor)
        return output[0].cpu().numpy().astype(np.float32)

    @staticmethod
    def _load_checkpoint(checkpoint: Path, embedder: FaceEmbedder) -> None:
        if not checkpoint.is_file():
            raise ModelLoadError(f"checkpoint not found at {checkpoint}; {TRAIN_HINT}")
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            # Training starts from the pretrained FaceNet, which carries the VGGFace2 classifier
            # (`logits`). It is not part of the embedding, and the network built here has no such
            # layer, so the strict load below would refuse it.
            weights = {
                name: value
                for name, value in saved["network"].items()
                if not name.startswith(CLASSIFIER_PREFIX)
            }
            cast(nn.Module, embedder.network).load_state_dict(weights)
        except Exception as error:  # torch's unpickler raises many types on a corrupt file
            raise ModelLoadError(
                f"{checkpoint} is not a valid FaceNet checkpoint ({type(error).__name__}); "
                f"{TRAIN_HINT}"
            ) from error
