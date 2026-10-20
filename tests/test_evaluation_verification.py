import numpy as np
import pytest
from pydantic import ValidationError

from facial_recognition_system.evaluation import (
    ThresholdCriterion,
    VerificationScores,
    choose_eer_threshold,
    choose_far_threshold,
    equal_error_rate,
    evaluate_verification,
    false_accept_rate,
    false_reject_rate,
    similarity_distributions,
    tar_at_far,
    verification_metrics,
)


def _scores(genuine: np.ndarray, impostor: np.ndarray) -> VerificationScores:
    return VerificationScores(
        scores=np.concatenate([genuine, impostor]),
        labels=np.concatenate([np.ones(len(genuine), int), np.zeros(len(impostor), int)]),
    )


@pytest.fixture
def overlapping_uniform() -> VerificationScores:
    # Genuine on [0.3, 1.0] and impostor on [0.0, 0.7]: FAR = FRR at 0.5, EER = 0.2 / 0.7.
    return _scores(np.linspace(0.3, 1.0, 10_001), np.linspace(0.0, 0.7, 10_001))


@pytest.fixture
def separated() -> VerificationScores:
    return _scores(np.array([0.8, 0.9, 0.95]), np.array([0.1, 0.2, 0.3]))


@pytest.mark.unit
def test_eer_of_overlapping_uniform_distributions(overlapping_uniform: VerificationScores) -> None:
    result = equal_error_rate(overlapping_uniform)

    assert result.eer == pytest.approx(0.2 / 0.7, abs=1e-3)
    assert result.threshold == pytest.approx(0.5, abs=1e-3)


@pytest.mark.unit
def test_eer_of_unit_gaussians_two_sigmas_apart() -> None:
    # Genuine N(1, 1) and impostor N(-1, 1): EER = Phi(-1) = 0.1587.
    rng = np.random.default_rng(0)
    scores = _scores(rng.normal(1, 1, 100_000), rng.normal(-1, 1, 100_000))

    assert equal_error_rate(scores).eer == pytest.approx(0.1587, abs=0.005)


@pytest.mark.unit
def test_tar_at_far_of_overlapping_uniform_distributions(
    overlapping_uniform: VerificationScores,
) -> None:
    # FAR = 1% at t = 0.693, where TAR = 1 - (0.693 - 0.3) / 0.7.
    expected = 1 - (0.693 - 0.3) / 0.7

    assert tar_at_far(overlapping_uniform, 0.01) == pytest.approx(expected, abs=1e-3)


@pytest.mark.unit
def test_separated_scores_give_perfect_metrics(separated: VerificationScores) -> None:
    threshold = choose_eer_threshold(separated).value
    metrics = verification_metrics(separated, threshold)

    assert metrics.roc_auc == 1.0
    assert metrics.eer == 0.0
    assert metrics.tar_at_far == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0


@pytest.mark.unit
def test_accuracy_and_f1_use_the_given_threshold(separated: VerificationScores) -> None:
    # At 0.92, two genuine pairs are rejected: 4 of 6 right, F1 = 2 / (2 + 0 + 2) * 2 / 2.
    metrics = verification_metrics(separated, threshold=0.92)

    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.f1 == pytest.approx(0.5)


@pytest.mark.unit
def test_far_and_frr_count_scores_at_the_threshold_as_accepted() -> None:
    scores = _scores(np.array([0.5, 0.6]), np.array([0.5, 0.4]))

    assert false_accept_rate(scores, 0.5) == 0.5
    assert false_reject_rate(scores, 0.5) == 0.0


@pytest.mark.unit
def test_far_threshold_is_the_smallest_with_far_at_most_the_target() -> None:
    # 100 impostors at 0.00 ... 0.99: FAR <= 1% allows only 0.99 to be accepted.
    scores = _scores(np.full(10, 1.0), np.arange(100) / 100)

    choice = choose_far_threshold(scores, max_far=0.01)

    assert choice.criterion == ThresholdCriterion.FAR_TARGET
    assert choice.value == pytest.approx(0.99)
    assert choice.far == pytest.approx(0.01)
    assert false_accept_rate(scores, 0.98) > 0.01
    assert choice.chosen_on == "val"


@pytest.mark.unit
def test_far_threshold_falls_back_to_eer_when_frr_is_too_high() -> None:
    # Half of the genuine scores sit among the impostors, so FAR <= 1% rejects ~50% of them.
    genuine = np.concatenate([np.full(50, 2.0), np.linspace(0.0, 0.98, 50)])
    scores = _scores(genuine, np.arange(100) / 100)

    choice = choose_far_threshold(scores, max_far=0.01, max_frr=0.20)

    assert choice.criterion == ThresholdCriterion.EER
    assert choice.value == choose_eer_threshold(scores).value


@pytest.mark.unit
def test_far_threshold_can_reject_every_impostor() -> None:
    scores = _scores(np.array([0.9, 0.95]), np.array([0.99]))

    choice = choose_far_threshold(scores, max_far=0.0, max_frr=1.0)

    assert choice.far == 0.0
    assert choice.value > 0.99


@pytest.mark.unit
def test_bootstrap_intervals_contain_the_value_and_are_reproducible(
    overlapping_uniform: VerificationScores,
) -> None:
    threshold = choose_eer_threshold(overlapping_uniform).value

    report = evaluate_verification(overlapping_uniform, threshold, n_resamples=50, seed=1)
    again = evaluate_verification(overlapping_uniform, threshold, n_resamples=50, seed=1)

    assert report == again
    for interval in (report.roc_auc, report.eer, report.tar_at_far, report.accuracy, report.f1):
        assert interval.low <= interval.value <= interval.high
    assert report.eer.high - report.eer.low < 0.05


@pytest.mark.unit
def test_similarity_distributions_split_genuine_and_impostor(
    separated: VerificationScores,
) -> None:
    distributions = similarity_distributions(separated)

    assert distributions.genuine == [0.8, 0.9, 0.95]
    assert distributions.impostor == [0.1, 0.2, 0.3]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scores", "labels"),
    [
        ([0.1, 0.2], [1]),
        ([0.1, 0.2], [1, 1]),
        ([0.1, 0.2], [1, 2]),
        ([0.1, float("nan")], [1, 0]),
        ([[0.1, 0.2]], [[1, 0]]),
    ],
)
def test_invalid_scores_are_rejected(scores: list[object], labels: list[object]) -> None:
    with pytest.raises(ValidationError):
        VerificationScores(scores=np.array(scores), labels=np.array(labels))
