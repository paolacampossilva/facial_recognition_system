"""Verification pairs of the contract (``lfw_<split>_pairs.csv``) turned into similarity scores."""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from facial_recognition_system.data import ImagePair
from facial_recognition_system.evaluation import VerificationScores
from facial_recognition_system.experiments.embeddings import EmbeddingSet


def read_pairs(path: Path) -> list[ImagePair]:
    """Reads a pairs CSV with the columns ``path_a,path_b,same`` (``same`` = 1 or 0)."""
    with path.open(newline="", encoding="utf-8") as file:
        return [
            ImagePair(path_a=row["path_a"], path_b=row["path_b"], same=row["same"] == "1")
            for row in csv.DictReader(file)
        ]


def pair_paths(pairs: Sequence[ImagePair]) -> list[str]:
    """Every photo used by the pairs, once, in a stable order."""
    return sorted({path for pair in pairs for path in (pair.path_a, pair.path_b)})


def pair_scores(embedding_set: EmbeddingSet, pairs: Sequence[ImagePair]) -> VerificationScores:
    """Cosine similarity of each pair, labeled 1 for the same person and 0 otherwise.

    The embeddings are L2-normalized, so the dot product is the cosine similarity.

    Raises:
        KeyError: A photo of a pair is not in ``embedding_set``.
    """
    index = {path: position for position, path in enumerate(embedding_set.paths)}
    first = embedding_set.embeddings[[index[pair.path_a] for pair in pairs]]
    second = embedding_set.embeddings[[index[pair.path_b] for pair in pairs]]
    return VerificationScores(
        scores=np.einsum("ij,ij->i", first, second).astype(np.float64),
        labels=np.array([int(pair.same) for pair in pairs]),
    )
