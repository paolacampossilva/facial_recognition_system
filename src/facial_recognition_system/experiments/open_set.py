"""E07 (comparison strategy) and E08 (open-set threshold) on the private gallery.

E07 fits every strategy of D6 on the gallery, chooses each one's threshold on
validation with the D8 criterion and keeps the one with the highest validation
F1 macro ("unknown" counted as a class). E08 takes that strategy and studies
its threshold: the FAR and FRR curve of validation, and the test metrics at the
D8 threshold and, for contrast, at the EER threshold. Every result is
aggregated; labels are anonymous IDs or "unknown".
"""

from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from facial_recognition_system.evaluation import (
    OpenSetAttempts,
    OpenSetMetrics,
    ThresholdChoice,
    choose_eer_threshold,
    choose_far_threshold,
    false_accept_rate,
    false_reject_rate,
    open_set_metrics,
)
from facial_recognition_system.experiments.config import DatasetConfig, EmbeddingConfig, ModelConfig
from facial_recognition_system.recognition import (
    GalleryStrategy,
    KNearestNeighbors,
    LinearSvm,
    NearestNeighbor,
    Prototype,
)

MAX_CURVE_POINTS: int = 100


class PrivateDataConfig(BaseModel):
    """Private crops and originals, both junctions to the vault (D10)."""

    model_config = ConfigDict(extra="forbid")

    crops_dir: Path = Path("faces/private")
    originals_dir: Path = Path("data/private")
    original_suffixes: list[str] = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


class StrategiesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knn_k: int = Field(default=3, ge=1)
    svm_c: list[float] = Field(default=[0.1, 1.0, 10.0], min_length=1)


class OpenSetThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_far: float = Field(default=0.01, gt=0, lt=1)
    max_frr: float = Field(default=0.20, gt=0, lt=1)


class OpenSetModelConfig(ModelConfig):
    # Best epoch of a fine-tuned experiment (checkpoints/<EXP>/best.pt); None keeps the
    # pretrained weights.
    checkpoint: Path | None = None


class OpenSetConfig(BaseModel):
    """Config of E07 and E08, which share the model, the private split and the embeddings."""

    model_config = ConfigDict(extra="forbid")

    selection_experiment: str = Field(default="E07", pattern=r"^E\d{2}$")
    threshold_experiment: str = Field(default="E08", pattern=r"^E\d{2}$")
    selection_hypothesis: str
    threshold_hypothesis: str
    # Best of E01 to E06 by validation EER (ADR "modelo escolhido").
    source_experiment: str = Field(pattern=r"^E\d{2}$")
    model: OpenSetModelConfig
    private: PrivateDataConfig = PrivateDataConfig()
    # LFW crops and split lists, for the 200 extra unknowns of each split.
    dataset: DatasetConfig = DatasetConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    strategies: StrategiesConfig = StrategiesConfig()
    threshold: OpenSetThresholdConfig = OpenSetThresholdConfig()
    seed: int = 42

    @model_validator(mode="after")
    def _check_distinct_experiments(self) -> Self:
        if self.selection_experiment == self.threshold_experiment:
            raise ValueError("selection and threshold experiments must have different IDs")
        return self


def load_open_set_config(path: Path) -> tuple[OpenSetConfig, dict[str, Any]]:
    """Reads and validates the E07/E08 YAML, returning it also as the raw mapping."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OpenSetConfig.model_validate(raw), raw


class LabeledEmbeddings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    embeddings: np.ndarray
    labels: list[str]


class StrategyResult(BaseModel):
    """One candidate of E07: its threshold and metrics on validation."""

    strategy: str
    params: dict[str, float]
    threshold: ThresholdChoice
    val: OpenSetMetrics


class StrategySelection(BaseModel):
    candidates: list[StrategyResult]
    chosen: StrategyResult


class CurvePoint(BaseModel):
    threshold: float
    far: float
    frr: float


class ThresholdStudy(BaseModel):
    """E08: the D8 threshold of the chosen strategy and its metrics on both splits."""

    strategy: str
    params: dict[str, float]
    threshold: ThresholdChoice
    eer_threshold: ThresholdChoice
    val_curve: list[CurvePoint]
    val: OpenSetMetrics
    test: OpenSetMetrics
    test_at_eer: OpenSetMetrics


def build_strategies(config: StrategiesConfig, seed: int) -> list[GalleryStrategy]:
    """Every candidate of D6, with one SVM per value of C."""
    strategies: list[GalleryStrategy] = [
        NearestNeighbor(),
        Prototype(),
        KNearestNeighbors(k=config.knn_k),
    ]
    strategies += [LinearSvm(c=c, seed=seed) for c in config.svm_c]
    return strategies


def strategy_params(strategy: GalleryStrategy) -> dict[str, float]:
    if isinstance(strategy, KNearestNeighbors):
        return {"k": strategy.k}
    if isinstance(strategy, LinearSvm):
        return {"c": strategy.c}
    return {}


def attempts_of(strategy: GalleryStrategy, queries: LabeledEmbeddings) -> OpenSetAttempts:
    """Matches the queries against the fitted gallery, keeping their true labels."""
    prediction = strategy.predict(queries.embeddings)
    return OpenSetAttempts(
        true_labels=queries.labels, predicted_labels=prediction.labels, scores=prediction.scores
    )


def select_strategy(
    strategies: list[GalleryStrategy],
    gallery: LabeledEmbeddings,
    val: LabeledEmbeddings,
    threshold: OpenSetThresholdConfig,
) -> StrategySelection:
    """E07: fits each strategy and keeps the highest validation F1 macro.

    Each strategy gets its own D8 threshold, because the rejection scores are
    not on the same scale (D6). Ties go to the strategy listed first, the simplest.

    Args:
        strategies: Unfitted candidates, from the simplest to the most complex.
        gallery: Embeddings of session 1 of the gallery participants.
        val: Validation attempts, with known and unknown people.
        threshold: FAR target and FRR limit of D8.

    Returns:
        Every candidate's validation result and the chosen one.
    """
    candidates: list[StrategyResult] = []
    for strategy in strategies:
        strategy.fit(gallery.embeddings, gallery.labels)
        attempts = attempts_of(strategy, val)
        choice = choose_far_threshold(
            attempts.rejection_scores(), threshold.max_far, threshold.max_frr
        )
        candidates.append(
            StrategyResult(
                strategy=strategy.name,
                params=strategy_params(strategy),
                threshold=choice,
                val=open_set_metrics(attempts, choice.value),
            )
        )

    # max() keeps the first of equal F1 values, so ties go to the simplest strategy.
    chosen = max(candidates, key=lambda candidate: candidate.val.f1_macro)
    return StrategySelection(candidates=candidates, chosen=chosen)


def study_threshold(
    strategy: GalleryStrategy,
    gallery: LabeledEmbeddings,
    val: LabeledEmbeddings,
    test: LabeledEmbeddings,
    threshold: OpenSetThresholdConfig,
) -> ThresholdStudy:
    """E08: chooses the D8 threshold on validation and reports it on test without readjusting.

    Args:
        strategy: The strategy chosen by E07, unfitted.
        gallery: Embeddings of the gallery.
        val: Validation attempts.
        test: Test attempts.
        threshold: FAR target and FRR limit of D8.

    Returns:
        The threshold, the validation FAR/FRR curve and the metrics of both
        splits, plus the test metrics at the EER threshold for comparison.
    """
    strategy.fit(gallery.embeddings, gallery.labels)
    val_attempts = attempts_of(strategy, val)
    test_attempts = attempts_of(strategy, test)
    val_rejection = val_attempts.rejection_scores()

    choice = choose_far_threshold(val_rejection, threshold.max_far, threshold.max_frr)
    eer_choice = choose_eer_threshold(val_rejection)

    candidates = np.unique(val_rejection.scores)
    if len(candidates) > MAX_CURVE_POINTS:
        candidates = np.quantile(candidates, np.linspace(0, 1, MAX_CURVE_POINTS))
    curve = [
        CurvePoint(
            threshold=float(value),
            far=false_accept_rate(val_rejection, float(value)),
            frr=false_reject_rate(val_rejection, float(value)),
        )
        for value in candidates
    ]

    return ThresholdStudy(
        strategy=strategy.name,
        params=strategy_params(strategy),
        threshold=choice,
        eer_threshold=eer_choice,
        val_curve=curve,
        val=open_set_metrics(val_attempts, choice.value),
        test=open_set_metrics(test_attempts, choice.value),
        test_at_eer=open_set_metrics(test_attempts, eer_choice.value),
    )
