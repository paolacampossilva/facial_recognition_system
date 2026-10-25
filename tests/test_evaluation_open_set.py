import numpy as np
import pytest
from pydantic import ValidationError

from facial_recognition_system.evaluation import (
    UNKNOWN_LABEL,
    OpenSetAttempts,
    open_set_metrics,
)


@pytest.fixture
def attempts() -> OpenSetAttempts:
    # Known P01 x3, known P02 x2 and unknown x3, decided at threshold 0.5:
    #   P01 right, P01 rejected, P01 taken for P02, P02 right, P02 right,
    #   unknown rejected, unknown rejected, unknown accepted as P01.
    return OpenSetAttempts(
        true_labels=["P01", "P01", "P01", "P02", "P02", *[UNKNOWN_LABEL] * 3],
        predicted_labels=["P01", "P01", "P02", "P02", "P02", "P01", "P02", "P01"],
        scores=np.array([0.9, 0.4, 0.7, 0.8, 0.6, 0.3, 0.2, 0.55]),
    )


@pytest.mark.unit
def test_open_set_rates_at_the_threshold(attempts: OpenSetAttempts) -> None:
    metrics = open_set_metrics(attempts, threshold=0.5)

    assert metrics.accuracy == pytest.approx(5 / 8)
    assert metrics.far == pytest.approx(1 / 3)
    assert metrics.frr == pytest.approx(1 / 5)
    assert metrics.misidentification_rate == pytest.approx(1 / 5)


@pytest.mark.unit
def test_counts_add_up_and_match_the_rates(attempts: OpenSetAttempts) -> None:
    metrics = open_set_metrics(attempts, threshold=0.5)
    counts = metrics.counts

    assert (counts.known_attempts, counts.unknown_attempts) == (5, 3)
    assert counts.known_correct + counts.known_misidentified + counts.known_rejected == 5
    assert counts.unknown_accepted + counts.unknown_rejected == 3
    assert (counts.known_correct, counts.known_misidentified, counts.known_rejected) == (3, 1, 1)
    assert (counts.unknown_accepted, counts.unknown_rejected) == (1, 2)
    assert metrics.far == pytest.approx(counts.unknown_accepted / counts.unknown_attempts)
    assert metrics.frr == pytest.approx(counts.known_rejected / counts.known_attempts)
    assert metrics.misidentification_rate == pytest.approx(
        counts.known_misidentified / counts.known_attempts
    )


@pytest.mark.unit
def test_f1_macro_averages_every_class_including_unknown(attempts: OpenSetAttempts) -> None:
    # Per class F1: P01 = 0.4, P02 = 0.8, unknown = 2/3.
    metrics = open_set_metrics(attempts, threshold=0.5)

    assert metrics.f1_macro == pytest.approx((0.4 + 0.8 + 2 / 3) / 3)


@pytest.mark.unit
def test_rates_without_unknown_attempts_are_none() -> None:
    attempts = OpenSetAttempts(
        true_labels=["P01", "P02"], predicted_labels=["P01", "P02"], scores=np.array([0.9, 0.8])
    )

    metrics = open_set_metrics(attempts, threshold=0.5)

    assert metrics.far is None
    assert metrics.frr == 0.0


@pytest.mark.unit
def test_rejection_scores_mark_known_attempts_as_genuine(attempts: OpenSetAttempts) -> None:
    scores = attempts.rejection_scores()

    assert scores.labels.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert scores.scores.tolist() == attempts.scores.tolist()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("true_labels", "predicted_labels", "scores"),
    [
        (["Ada_Lovelace"], ["P01"], [0.9]),
        (["P01"], ["Alan Turing"], [0.9]),
        (["P01", "P02"], ["P01"], [0.9, 0.8]),
        (["P01"], [UNKNOWN_LABEL], [0.9]),
        ([], [], []),
    ],
)
def test_invalid_attempts_are_rejected(
    true_labels: list[str], predicted_labels: list[str], scores: list[float]
) -> None:
    with pytest.raises(ValidationError):
        OpenSetAttempts(
            true_labels=true_labels, predicted_labels=predicted_labels, scores=np.array(scores)
        )


@pytest.mark.unit
def test_rejection_message_does_not_echo_the_names() -> None:
    with pytest.raises(ValidationError) as error:
        OpenSetAttempts(true_labels=["Ada_Lovelace"], predicted_labels=["P01"], scores=[0.9])

    assert "Ada_Lovelace" not in str(error.value)
