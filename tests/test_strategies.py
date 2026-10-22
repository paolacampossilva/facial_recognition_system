import numpy as np
import pytest

from facial_recognition_system.recognition import (
    GalleryStrategy,
    KNearestNeighbors,
    LinearSvm,
    NearestNeighbor,
    Prototype,
)

DIRECTIONS: dict[str, np.ndarray] = {
    "P01": np.array([1.0, 0.0, 0.0, 0.0]),
    "P02": np.array([0.0, 1.0, 0.0, 0.0]),
    "P03": np.array([0.0, 0.0, 1.0, 0.0]),
}


def _normalized(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _gallery(photos_per_person: int = 6) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for label, direction in DIRECTIONS.items():
        for _ in range(photos_per_person):
            rows.append(direction + rng.normal(scale=0.1, size=4))
            labels.append(label)
    return _normalized(np.array(rows)), labels


ALL_STRATEGIES: list[GalleryStrategy] = [
    NearestNeighbor(),
    Prototype(),
    KNearestNeighbors(k=3),
    LinearSvm(c=1.0),
]


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
def test_strategy_matches_queries_to_the_right_person(strategy: GalleryStrategy) -> None:
    embeddings, labels = _gallery()
    queries = _normalized(np.stack(list(DIRECTIONS.values())) + 0.05)

    prediction = strategy.fit(embeddings, labels).predict(queries)

    assert prediction.labels == list(DIRECTIONS)
    assert prediction.scores.shape == (3,)


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
def test_stranger_scores_lower_than_a_known_person(strategy: GalleryStrategy) -> None:
    embeddings, labels = _gallery()
    known = _normalized(np.array([[1.0, 0.05, 0.0, 0.0]]))
    stranger = np.array([[0.0, 0.0, 0.0, 1.0]])

    strategy.fit(embeddings, labels)

    assert strategy.predict(stranger).scores[0] < strategy.predict(known).scores[0]


@pytest.mark.unit
def test_prototype_score_is_the_cosine_with_the_renormalized_mean() -> None:
    embeddings = _normalized(np.array([[1.0, 1.0], [1.0, -1.0]]))

    prediction = Prototype().fit(embeddings, ["P01", "P01"]).predict(np.array([[1.0, 0.0]]))

    assert prediction.scores[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_knn_weights_votes_by_similarity() -> None:
    # Two weak P02 neighbors lose to one P01 neighbor that is much closer.
    gallery = _normalized(np.array([[1.0, 0.0], [0.0, 1.0], [-0.1, 1.0]]))
    query = _normalized(np.array([[1.0, 0.05]]))

    prediction = KNearestNeighbors(k=3).fit(gallery, ["P01", "P02", "P02"]).predict(query)

    assert prediction.labels == ["P01"]
    assert prediction.scores[0] == pytest.approx(float(query[0] @ gallery[0]))


@pytest.mark.unit
def test_svm_score_is_a_probability() -> None:
    embeddings, labels = _gallery()

    prediction = LinearSvm(c=1.0).fit(embeddings, labels).predict(embeddings)

    assert ((prediction.scores > 0) & (prediction.scores <= 1)).all()
