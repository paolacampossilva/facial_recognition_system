"""Batch embedding of face crops and the ``.npz`` format of the contract between slices.

Each file holds ``embeddings`` (float32, N x D, L2-normalized), ``paths`` and
``labels`` (strings, the label being the identity folder) and ``model_id``. The
evaluation reads them without assuming the dimension D.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from facial_recognition_system.models import FaceEmbedder
from facial_recognition_system.preprocessing import load_image


class EmbeddingSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    embeddings: np.ndarray
    paths: list[str]
    labels: list[str]
    model_id: str


def read_split(path: Path) -> list[str]:
    """Reads a split list: one crop path per line, relative to the crops folder."""
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def identity_of(path: str) -> str:
    """The identity is the parent folder: ``George_W_Bush/George_W_Bush_0001.png``."""
    return path.rsplit("/", 1)[0]


def embed_crops(
    embedder: FaceEmbedder,
    crops_dir: Path,
    paths: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> EmbeddingSet:
    """Embeds every crop in ``paths``, in order, in batches.

    Args:
        embedder: Wrapper of the backbone, already on ``device`` and in eval mode.
        crops_dir: Folder the paths are relative to.
        paths: Crop paths, relative to ``crops_dir``.
        batch_size: Number of crops per forward pass.
        device: Device the embedder runs on.

    Returns:
        The embeddings, in the same order as ``paths``, with the labels.
    """
    batches = []
    for start in tqdm(range(0, len(paths), batch_size), desc="embeddings", unit="batch"):
        faces = torch.stack(
            [
                embedder.preprocess(load_image(crops_dir / path))
                for path in paths[start : start + batch_size]
            ]
        )
        with torch.inference_mode():
            batches.append(embedder(faces.to(device)).float().cpu().numpy())

    embeddings = (
        np.concatenate(batches)
        if batches
        else np.empty((0, embedder.embedding_dim), dtype=np.float32)
    )
    return EmbeddingSet(
        embeddings=embeddings.astype(np.float32, copy=False),
        paths=list(paths),
        labels=[identity_of(path) for path in paths],
        model_id=embedder.model_id,
    )


def save_embeddings(embedding_set: EmbeddingSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        embeddings=embedding_set.embeddings,
        paths=np.array(embedding_set.paths, dtype=str),
        labels=np.array(embedding_set.labels, dtype=str),
        model_id=np.array(embedding_set.model_id),
    )


def load_embeddings(path: Path) -> EmbeddingSet:
    with np.load(path, allow_pickle=False) as data:
        return EmbeddingSet(
            embeddings=data["embeddings"],
            paths=data["paths"].tolist(),
            labels=data["labels"].tolist(),
            model_id=str(data["model_id"]),
        )
