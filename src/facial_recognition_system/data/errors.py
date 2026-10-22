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


class UnknownSplitError(SplitError):
    """Photos refer to splits that the split order does not list."""

    def __init__(self, splits: list[str]) -> None:
        super().__init__(f"splits missing from the split order: {splits}")
        self.splits = splits


class NotEnoughNegativePairsError(SplitError):
    """A split cannot form as many distinct negative pairs as positive ones."""

    def __init__(self, split: str, available: int, required: int) -> None:
        super().__init__(
            f"{split} split allows {available} distinct negative pairs, {required} are needed"
        )
        self.split = split
        self.available = available
        self.required = required


class NotEnoughParticipantsError(SplitError):
    """The private dataset has too few participants for a gallery plus the unknowns."""

    def __init__(self, available: int, required: int) -> None:
        super().__init__(
            f"private dataset has {available} participants, at least {required} are needed"
        )
        self.available = available
        self.required = required


class IncompleteParticipantError(SplitError):
    """A participant lacks the photos of a session.

    The message only carries the anonymous ID, never a name.
    """

    def __init__(self, participant: str, session: str, photos: int, required: int) -> None:
        super().__init__(
            f"{participant} has {photos} photo(s) in {session}, at least {required} are needed"
        )
        self.participant = participant
        self.session = session
        self.photos = photos
        self.required = required


class NonAnonymousFolderError(SplitError):
    """Folders of the private dataset are not anonymous IDs like ``P01``.

    Only the count goes to the message, so a name never reaches the logs.
    """

    def __init__(self, count: int) -> None:
        super().__init__(f"{count} private folder(s) are not anonymous IDs like 'P01'")
        self.count = count
