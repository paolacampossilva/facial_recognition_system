"""Downloads the public datasets used in the experiments into data/public/.

LFW is fetched from the figshare copy that scikit-learn also uses, because the
official site (vis-www.cs.umass.edu/lfw) is offline. Every archive is verified
against a pinned SHA-256 before extraction, and nothing is redistributed by
this repository.

Usage:
    python scripts/download_datasets.py [--data-dir data/public] [--force]
"""

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, HttpUrl

DEFAULT_DATA_DIR: Path = Path("data/public")
CHUNK_SIZE: int = 1024 * 1024
REQUEST_TIMEOUT_SECONDS: int = 60


class RemoteArchive(BaseModel):
    """A dataset archive hosted at a fixed URL with a known checksum."""

    model_config = ConfigDict(frozen=True)

    name: str
    url: HttpUrl
    filename: str
    sha256: str
    extracted_dir: str


LFW: RemoteArchive = RemoteArchive(
    name="LFW (Labeled Faces in the Wild)",
    url=HttpUrl("https://ndownloader.figshare.com/files/5976018"),
    filename="lfw.tgz",
    sha256="055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
    extracted_dir="lfw",
)

Downloader = Callable[[str, Path], None]


class DatasetDownloadError(Exception):
    """Base error for a public dataset that could not be prepared."""


class ChecksumMismatchError(DatasetDownloadError):
    """The downloaded archive does not match the pinned SHA-256."""

    def __init__(self, path: Path, expected: str, actual: str) -> None:
        super().__init__(f"{path} has SHA-256 {actual}, expected {expected}")
        self.path = path
        self.expected = expected
        self.actual = actual


def sha256_of(path: Path) -> str:
    """Computes the SHA-256 hex digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """Streams a URL to disk through a temporary .part file.

    Args:
        url: Address of the file.
        destination: Final path; it only appears once the download completes.
    """
    partial = destination.with_name(f"{destination.name}.part")
    with (
        urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response,
        partial.open("wb") as file,
    ):
        shutil.copyfileobj(response, file, CHUNK_SIZE)
    partial.replace(destination)


def fetch_dataset(
    archive: RemoteArchive,
    data_dir: Path,
    force: bool = False,
    downloader: Downloader = download_file,
) -> Path:
    """Downloads, verifies and extracts a dataset archive.

    Skips the download when the archive is already on disk with the right
    checksum, and skips everything when the dataset is already extracted.

    Args:
        archive: The archive to fetch.
        data_dir: Folder that receives the archive and the extracted dataset.
        force: Downloads and extracts again even if the files already exist.
        downloader: Function that saves a URL to a path (injected in tests).

    Returns:
        Path of the extracted dataset folder.

    Raises:
        ChecksumMismatchError: The downloaded archive is corrupted or changed upstream.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / archive.filename
    extracted_path = data_dir / archive.extracted_dir

    if extracted_path.is_dir() and not force:
        print(f"{archive.name}: already extracted at {extracted_path}, skipping")
        return extracted_path

    has_valid_archive = archive_path.is_file() and sha256_of(archive_path) == archive.sha256
    if force or not has_valid_archive:
        print(f"{archive.name}: downloading {archive.url}")
        downloader(str(archive.url), archive_path)

    actual_sha256 = sha256_of(archive_path)
    if actual_sha256 != archive.sha256:
        archive_path.unlink()
        raise ChecksumMismatchError(archive_path, archive.sha256, actual_sha256)

    if extracted_path.exists():
        shutil.rmtree(extracted_path)

    print(f"{archive.name}: extracting to {extracted_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir, filter="data")

    return extracted_path


def main(argv: Sequence[str] | None = None, downloader: Downloader = download_file) -> int:
    """Fetches every public dataset and reports failures.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.
        downloader: Function that saves a URL to a path (injected in tests).

    Returns:
        0 on success, 1 when any dataset could not be prepared.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="download and extract again")
    args = parser.parse_args(argv)

    try:
        extracted_path = fetch_dataset(LFW, args.data_dir, force=args.force, downloader=downloader)
    except (DatasetDownloadError, tarfile.TarError, OSError) as error:
        print(f"Failed to prepare {LFW.name}: {error}", file=sys.stderr)
        return 1

    identity_count = sum(1 for path in extracted_path.iterdir() if path.is_dir())
    print(f"{LFW.name}: ready at {extracted_path} ({identity_count} identities)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
