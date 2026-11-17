"""SQLite repository of people and embeddings (decision D12, specs 002, 003 and 006).

One connection serves the whole repository: an in-memory database only exists inside its
connection, and the API routes run on a thread pool, so the connection is shared across threads
behind a lock. The PRAGMAs of D12 are set once, on that connection.
"""

import os
import sqlite3
import threading
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.core.errors import (
    CorruptEmbeddingError,
    DatabaseUnavailableError,
    DuplicatePersonError,
    PersonNotFoundError,
)
from app.core.models import DeletionResult, PersonSummary
from app.storage.models import Gallery, NewPerson

MEMORY_DATABASE: str = ":memory:"
# 2: AUTOINCREMENT, so the id of a deleted person or embedding is never given to another one.
SCHEMA_VERSION: int = 2
PRIVATE_FILE_MODE: int = 0o600  # biometric data: only the owner reads the file
PRIVATE_FOLDER_MODE: int = 0o700

SCHEMA: str = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons (id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS embeddings_person_id ON embeddings (person_id);
CREATE INDEX IF NOT EXISTS embeddings_model_id ON embeddings (model_id);
"""


def name_key(name: str) -> str:
    """The form of a name used to tell whether two names are the same person's.

    It ignores case and accent case ("Ágata" and "ágata" give the same key); SQLite's own
    NOCASE collation would only fold ASCII letters.
    """
    return unicodedata.normalize("NFC", name).casefold()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_database_folder(database_path: Path | str) -> None:
    """Makes sure the folder of the database exists, creating it private when it does not.

    Raises:
        DatabaseUnavailableError: The folder cannot be created or used. The usual cause is a
            `data/private` that links to a vault which is closed: the link is there, but what it
            points to is not, and the operating system refuses to create it again.
    """
    folder = Path(database_path).parent
    try:
        folder.mkdir(mode=PRIVATE_FOLDER_MODE, parents=True, exist_ok=True)
    except OSError as error:
        raise DatabaseUnavailableError(
            f"the folder of the database is not available: {folder} ({type(error).__name__}); "
            f"if it is a link to the vault, open the vault first"
        ) from error


class SqlitePersonRepository:
    """Stores people with their embeddings and reads them back for recognition."""

    def __init__(self, database_path: Path | str) -> None:
        """Opens the database, creating the file, its folder and the schema on first use.

        Args:
            database_path: A file path, or `":memory:"` for a throwaway database (tests).

        Raises:
            DatabaseUnavailableError: The folder or the file of the database cannot be used.
        """
        self._lock = threading.Lock()
        target = str(database_path)
        if target != MEMORY_DATABASE:
            ensure_database_folder(target)
            try:
                # Create the file private before SQLite does: SQLite would create it with the
                # umask (usually readable by others), and its journal file copies its mode.
                os.close(os.open(target, os.O_CREAT | os.O_RDWR, PRIVATE_FILE_MODE))
            except OSError as error:
                raise DatabaseUnavailableError(
                    f"the database file cannot be created: {target} ({type(error).__name__})"
                ) from error
        self._connection = sqlite3.connect(target, check_same_thread=False)
        # Without these two PRAGMAs the cascade is ignored and deleted content stays in the file.
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA secure_delete = ON")
        new_file = self._connection.execute("PRAGMA user_version").fetchone()[0] == 0
        self._connection.executescript(SCHEMA)
        # Stamped only on a new file: `CREATE TABLE IF NOT EXISTS` does not change an older one,
        # so an older file keeps its own version instead of claiming the current schema.
        if new_file:
            self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        if target != MEMORY_DATABASE:
            os.chmod(target, PRIVATE_FILE_MODE)  # also tightens a file an older version created

    def add_person(self, new: NewPerson) -> PersonSummary:
        """Stores a person and all their embeddings in one transaction.

        Raises:
            DuplicatePersonError: The name is already used (ignoring case and accents).
        """
        created_at = _now()
        with self._lock, self._connection:
            try:
                cursor = self._connection.execute(
                    "INSERT INTO persons (name, name_key, created_at) VALUES (?, ?, ?)",
                    (new.name, name_key(new.name), created_at),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicatePersonError(new.name, "name") from error
            person_id = int(cursor.lastrowid or 0)
            self._connection.executemany(
                "INSERT INTO embeddings (person_id, vector, dimension, model_id, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (person_id, vector.tobytes(), vector.size, new.model_id, created_at)
                    for vector in new.vectors
                ],
            )
        return PersonSummary(id=person_id, name=new.name, embedding_count=len(new.vectors))

    def list_persons(self) -> list[PersonSummary]:
        """Returns every person in enrollment order, with how many embeddings each has."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT p.id, p.name, COUNT(e.id) FROM persons p"
                " LEFT JOIN embeddings e ON e.person_id = p.id GROUP BY p.id ORDER BY p.id"
            ).fetchall()
        return [PersonSummary(id=row[0], name=row[1], embedding_count=row[2]) for row in rows]

    def gallery(self, model_id: str) -> Gallery:
        """Returns the embeddings made by `model_id` (vectors of other models are ignored).

        Raises:
            CorruptEmbeddingError: A stored vector does not match its recorded dimension, or the
                vectors of the model do not share one dimension.
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT e.id, p.name, e.vector, e.dimension FROM embeddings e"
                " JOIN persons p ON p.id = e.person_id WHERE e.model_id = ? ORDER BY e.id",
                (model_id,),
            ).fetchall()
        if not rows:
            return Gallery(embeddings=np.empty((0, 0), dtype=np.float32), labels=[])
        vectors = [
            self._decode(embedding_id, blob, dimension) for embedding_id, _, blob, dimension in rows
        ]
        if len({vector.size for vector in vectors}) != 1:
            raise CorruptEmbeddingError(rows[0][0])
        return Gallery(embeddings=np.stack(vectors), labels=[row[1] for row in rows])

    def delete_person(self, person_id: int) -> DeletionResult:
        """Deletes a person; the cascade removes their embeddings in the same transaction.

        Raises:
            PersonNotFoundError: No person has this id; nothing changes.
        """
        with self._lock, self._connection:
            deleted_embeddings = self._connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
            ).fetchone()[0]
            cursor = self._connection.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            if cursor.rowcount == 0:
                raise PersonNotFoundError(person_id)
        return DeletionResult(person_id=person_id, deleted_embeddings=deleted_embeddings)

    def close(self) -> None:
        """Closes the connection."""
        with self._lock:
            self._connection.close()

    @staticmethod
    def _decode(embedding_id: int, blob: bytes, dimension: int) -> np.ndarray:
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != dimension:
            raise CorruptEmbeddingError(embedding_id)
        return vector
