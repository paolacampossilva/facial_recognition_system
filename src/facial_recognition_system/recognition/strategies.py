"""Gallery comparison strategies of decision D6.

Each strategy is fitted on the gallery embeddings (L2-normalized) and, for each
query, returns the gallery identity it picks and a rejection score. The
open-set decision ("unknown" below a threshold) is not taken here: the
threshold is chosen on validation (D8) and applied by the caller.
"""

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

# Platt calibration folds; fewer when a person has fewer gallery photos.
MAX_CALIBRATION_FOLDS: int = 5


class GalleryPrediction(BaseModel):
    """The identity each query was matched to and how confident the match is."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    labels: list[str]
    scores: np.ndarray


class GalleryStrategy(ABC):
    """A way to match a query embedding against a gallery of known people."""

    name: str

    @abstractmethod
    def fit(self, embeddings: np.ndarray, labels: Sequence[str]) -> Self:
        """Enrolls the gallery: one row per photo, labeled with its identity."""

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> GalleryPrediction:
        """Picks a gallery identity for each row, with its rejection score."""


class NearestNeighbor(GalleryStrategy):
    """Identity of the most similar gallery photo; the score is that similarity."""

    name = "nearest_neighbor"

    def fit(self, embeddings: np.ndarray, labels: Sequence[str]) -> Self:
        self._embeddings = embeddings
        self._labels = np.array(labels)
        return self

    def predict(self, embeddings: np.ndarray) -> GalleryPrediction:
        similarities = embeddings @ self._embeddings.T
        best = similarities.argmax(axis=1)
        return GalleryPrediction(
            labels=self._labels[best].tolist(),
            scores=similarities[np.arange(len(embeddings)), best].astype(np.float64),
        )


class Prototype(GalleryStrategy):
    """Compares the query with each person's mean embedding, renormalized."""

    name = "prototype"

    def fit(self, embeddings: np.ndarray, labels: Sequence[str]) -> Self:
        label_array = np.array(labels)
        self._identities = np.array(sorted(set(labels)))
        prototypes = np.stack(
            [embeddings[label_array == identity].mean(axis=0) for identity in self._identities]
        )
        self._prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
        return self

    def predict(self, embeddings: np.ndarray) -> GalleryPrediction:
        similarities = embeddings @ self._prototypes.T
        best = similarities.argmax(axis=1)
        return GalleryPrediction(
            labels=self._identities[best].tolist(),
            scores=similarities[np.arange(len(embeddings)), best].astype(np.float64),
        )


class KNearestNeighbors(GalleryStrategy):
    """Votes of the k most similar gallery photos, weighted by similarity.

    The score is the mean similarity of the winning identity's neighbors.
    """

    name = "knn"

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, embeddings: np.ndarray, labels: Sequence[str]) -> Self:
        self._embeddings = embeddings
        self._labels = np.array(labels)
        return self

    def predict(self, embeddings: np.ndarray) -> GalleryPrediction:
        similarities = embeddings @ self._embeddings.T
        k = min(self.k, similarities.shape[1])
        neighbors = np.argsort(-similarities, axis=1)[:, :k]

        labels: list[str] = []
        scores: list[float] = []
        for row, indices in enumerate(neighbors):
            neighbor_labels = self._labels[indices]
            neighbor_similarities = similarities[row, indices]
            votes = {
                label: neighbor_similarities[neighbor_labels == label].sum()
                for label in neighbor_labels
            }
            winner = max(votes, key=lambda label: votes[label])
            labels.append(str(winner))
            scores.append(float(neighbor_similarities[neighbor_labels == winner].mean()))
        return GalleryPrediction(labels=labels, scores=np.array(scores))


class LinearSvm(GalleryStrategy):
    """Linear SVM over the gallery; the score is the highest Platt-calibrated probability."""

    name = "svm"

    def __init__(self, c: float = 1.0, seed: int = 42) -> None:
        self.c = c
        self.seed = seed

    def fit(self, embeddings: np.ndarray, labels: Sequence[str]) -> Self:
        smallest_class = min(Counter(labels).values())
        folds = max(2, min(MAX_CALIBRATION_FOLDS, smallest_class))
        self._model = CalibratedClassifierCV(
            SVC(kernel="linear", C=self.c, random_state=self.seed),
            method="sigmoid",
            cv=folds,
            ensemble=False,
        )
        self._model.fit(embeddings, list(labels))
        return self

    def predict(self, embeddings: np.ndarray) -> GalleryPrediction:
        probabilities = self._model.predict_proba(embeddings)
        best = probabilities.argmax(axis=1)
        return GalleryPrediction(
            labels=self._model.classes_[best].tolist(),
            scores=probabilities[np.arange(len(embeddings)), best].astype(np.float64),
        )
