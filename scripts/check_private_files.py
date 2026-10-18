"""Pre-commit hook that blocks private biometric data from being committed.

Photos, videos, embeddings, checkpoints, fitted classifiers and databases may
derive from the private dataset (LGPD), so they are rejected everywhere. PNG
figures are accepted only in the folders that hold documentation and aggregated
plots, and only data/README.md and data/splits/ may be versioned under data/.

Usage:
    python scripts/check_private_files.py FILE [FILE ...]
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import PurePosixPath

BLOCKED_SUFFIXES: frozenset[str] = frozenset(
    {
        # Photos
        ".jpg",
        ".jpeg",
        ".heic",
        ".heif",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
        # Videos
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        # Model weights
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".safetensors",
        # Embeddings and fitted classifiers
        ".npy",
        ".npz",
        ".pkl",
        ".joblib",
        # Databases
        ".db",
        ".sqlite",
        ".sqlite3",
    }
)
FIGURE_SUFFIXES: frozenset[str] = frozenset({".png"})
FIGURE_DIRS: tuple[PurePosixPath, ...] = (
    PurePosixPath("docs"),
    PurePosixPath("experiments/figures"),
)
PRIVATE_DIR_NAMES: frozenset[str] = frozenset({"faces", "embeddings", "checkpoints", "consent"})
DATA_DIR: PurePosixPath = PurePosixPath("data")
ALLOWED_DATA_PATHS: tuple[PurePosixPath, ...] = (
    PurePosixPath("data/README.md"),
    PurePosixPath("data/splits"),
)


def _is_under(path: PurePosixPath, parents: Sequence[PurePosixPath]) -> bool:
    return any(path == parent or path.is_relative_to(parent) for parent in parents)


def find_violation(path: str) -> str | None:
    """Explains why a file must not be committed.

    Args:
        path: Repository-relative path of the staged file.

    Returns:
        The reason the file is blocked, or None when it may be committed.
    """
    file_path = PurePosixPath(path.replace("\\", "/"))
    suffix = file_path.suffix.lower()
    folder_names = set(file_path.parts[:-1])

    private_folders = folder_names & PRIVATE_DIR_NAMES
    if private_folders:
        return f"inside private folder '{sorted(private_folders)[0]}/'"

    if file_path.is_relative_to(DATA_DIR) and not _is_under(file_path, ALLOWED_DATA_PATHS):
        return "only data/README.md and data/splits/ may be versioned under data/"

    if suffix in BLOCKED_SUFFIXES:
        return f"'{suffix}' files may contain photos or biometric data"

    if suffix in FIGURE_SUFFIXES and not _is_under(file_path, FIGURE_DIRS):
        return "PNG files are only allowed in docs/ and experiments/figures/"

    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Checks the given files and reports every blocked one.

    Args:
        argv: File paths passed by pre-commit. Defaults to sys.argv.

    Returns:
        0 when every file may be committed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    violations = [(path, find_violation(path)) for path in args.filenames]
    blocked = [(path, reason) for path, reason in violations if reason is not None]

    for path, reason in blocked:
        print(f"{path}: {reason}", file=sys.stderr)

    if blocked:
        print(
            f"\nBlocked {len(blocked)} file(s) that may contain private data (LGPD). "
            "Remove them from the commit with 'git restore --staged <file>'.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
