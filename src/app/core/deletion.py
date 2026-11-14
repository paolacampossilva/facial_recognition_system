"""Deletion of an enrolled person and everything tied to them (spec 006)."""

import logging

from app.core.models import DeletionResult
from app.storage.repository import SqlitePersonRepository

logger = logging.getLogger(__name__)


class DeletionService:
    """Deletes a person. The repository does it in one transaction, with the cascade on."""

    def __init__(self, repository: SqlitePersonRepository) -> None:
        self._repository = repository

    def delete(self, person_id: int) -> DeletionResult:
        """Deletes the person, their embeddings and anything else tied to the identity.

        Photos are never stored (decision D12), so there are no files to remove, and recognition
        reads the repository on every call, so there is no cache to invalidate.

        Raises:
            PersonNotFoundError: No person has this id; nothing changes.
        """
        result = self._repository.delete_person(person_id)
        logger.info(
            "person deleted",
            extra={"person_id": result.person_id, "deleted_embeddings": result.deleted_embeddings},
        )
        return result
