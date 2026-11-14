"""Models at the edge of the storage: what goes in and what comes out as arrays."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.models import MAX_NAME_LENGTH


class NewPerson(BaseModel):
    """A person to store: a name and one embedding per enrolled photo."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    model_id: str = Field(min_length=1)
    vectors: list[np.ndarray] = Field(min_length=1)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: object) -> object:
        """Remove the spaces at both ends before the length check."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("vectors")
    @classmethod
    def check_vectors(cls, vectors: list[np.ndarray]) -> list[np.ndarray]:
        """Keep float32 vectors that are one-dimensional, finite and of the same size."""
        converted = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        for vector in converted:
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("each embedding must be a non-empty finite 1-D vector")
        if len({vector.size for vector in converted}) != 1:
            raise ValueError("all embeddings of a person must have the same dimension")
        return converted


class Gallery(BaseModel):
    """The embeddings of one model, one row per stored photo, with the name of each owner.

    The shape matches `GalleryStrategy.fit(embeddings, labels)`. An empty gallery has a
    `(0, 0)` matrix and no labels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    embeddings: np.ndarray
    labels: list[str]
