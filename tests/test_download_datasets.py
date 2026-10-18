import hashlib
import io
import shutil
import tarfile
from pathlib import Path

import pytest
from pydantic import HttpUrl

from scripts.download_datasets import (
    ChecksumMismatchError,
    RemoteArchive,
    fetch_dataset,
    main,
)


def _build_archive(path: Path, members: dict[str, bytes]) -> str:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeDownloader:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls = 0

    def __call__(self, url: str, destination: Path) -> None:
        self.calls += 1
        shutil.copyfile(self.source, destination)


@pytest.fixture
def remote(tmp_path: Path) -> tuple[RemoteArchive, FakeDownloader]:
    source = tmp_path / "remote" / "lfw.tgz"
    source.parent.mkdir()
    sha256 = _build_archive(source, {"lfw/Ada_Lovelace/Ada_Lovelace_0001.jpg": b"fake image"})
    archive = RemoteArchive(
        name="fake LFW",
        url=HttpUrl("https://example.invalid/lfw.tgz"),
        filename="lfw.tgz",
        sha256=sha256,
        extracted_dir="lfw",
    )
    return archive, FakeDownloader(source)


@pytest.mark.unit
def test_fetch_downloads_verifies_and_extracts(
    tmp_path: Path, remote: tuple[RemoteArchive, FakeDownloader]
) -> None:
    archive, downloader = remote
    data_dir = tmp_path / "data"

    extracted = fetch_dataset(archive, data_dir, downloader=downloader)

    assert extracted == data_dir / "lfw"
    assert (extracted / "Ada_Lovelace" / "Ada_Lovelace_0001.jpg").read_bytes() == b"fake image"
    assert downloader.calls == 1


@pytest.mark.unit
def test_fetch_skips_when_already_extracted(
    tmp_path: Path, remote: tuple[RemoteArchive, FakeDownloader]
) -> None:
    archive, downloader = remote
    data_dir = tmp_path / "data"
    fetch_dataset(archive, data_dir, downloader=downloader)

    fetch_dataset(archive, data_dir, downloader=downloader)

    assert downloader.calls == 1


@pytest.mark.unit
def test_fetch_reuses_valid_archive_without_downloading(
    tmp_path: Path, remote: tuple[RemoteArchive, FakeDownloader]
) -> None:
    archive, downloader = remote
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copyfile(downloader.source, data_dir / archive.filename)

    fetch_dataset(archive, data_dir, downloader=downloader)

    assert downloader.calls == 0
    assert (data_dir / "lfw" / "Ada_Lovelace").is_dir()


@pytest.mark.unit
def test_force_downloads_again(
    tmp_path: Path, remote: tuple[RemoteArchive, FakeDownloader]
) -> None:
    archive, downloader = remote
    data_dir = tmp_path / "data"
    fetch_dataset(archive, data_dir, downloader=downloader)

    fetch_dataset(archive, data_dir, force=True, downloader=downloader)

    assert downloader.calls == 2


@pytest.mark.unit
def test_checksum_mismatch_raises_and_removes_archive(
    tmp_path: Path, remote: tuple[RemoteArchive, FakeDownloader]
) -> None:
    archive, downloader = remote
    tampered = archive.model_copy(update={"sha256": "0" * 64})
    data_dir = tmp_path / "data"

    with pytest.raises(ChecksumMismatchError):
        fetch_dataset(tampered, data_dir, downloader=downloader)

    assert not (data_dir / archive.filename).exists()
    assert not (data_dir / "lfw").exists()


@pytest.mark.unit
def test_archive_with_path_traversal_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "evil.tgz"
    sha256 = _build_archive(source, {"../outside.txt": b"escaped"})
    archive = RemoteArchive(
        name="evil",
        url=HttpUrl("https://example.invalid/evil.tgz"),
        filename="evil.tgz",
        sha256=sha256,
        extracted_dir="lfw",
    )
    data_dir = tmp_path / "data"

    with pytest.raises(tarfile.FilterError):
        fetch_dataset(archive, data_dir, downloader=FakeDownloader(source))

    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.unit
def test_main_returns_one_when_download_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_download(url: str, destination: Path) -> None:
        raise OSError("network unreachable")

    exit_code = main(["--data-dir", str(tmp_path / "data")], downloader=failing_download)

    assert exit_code == 1
    assert "network unreachable" in capsys.readouterr().err
