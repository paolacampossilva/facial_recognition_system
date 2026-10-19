"""Errors raised when the public splits cannot be built."""


class SplitError(Exception):
    """Base error for a dataset that cannot be split as configured."""


class NotEnoughUnknownsError(SplitError):
    """A split has fewer single-photo identities than the unknowns it needs."""

    def __init__(self, split: str, available: int, required: int) -> None:
        super().__init__(
            f"{split} split has {available} single-photo identities, "
            f"{required} are needed as unknowns"
        )
        self.split = split
        self.available = available
        self.required = required


class NotEnoughNegativePairsError(SplitError):
    """A split cannot form as many distinct negative pairs as positive ones."""

    def __init__(self, split: str, available: int, required: int) -> None:
        super().__init__(
            f"{split} split allows {available} distinct negative pairs, {required} are needed"
        )
        self.split = split
        self.available = available
        self.required = required
