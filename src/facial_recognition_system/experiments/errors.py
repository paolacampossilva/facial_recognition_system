"""Errors raised when an experiment cannot be evaluated."""


class ExperimentError(Exception):
    """Base error for an experiment whose files do not follow the contract."""


class MissingEmbeddingsError(ExperimentError):
    """The results JSON lists no embeddings file for a split the evaluation needs."""

    def __init__(self, experiment: str, split: str) -> None:
        super().__init__(
            f"{experiment} has no {split} embeddings in its results JSON; "
            f"run scripts/extract_embeddings.py or scripts/train.py first"
        )
        self.experiment = experiment
        self.split = split
