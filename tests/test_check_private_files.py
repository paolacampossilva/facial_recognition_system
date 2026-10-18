import pytest

from scripts.check_private_files import find_violation, main


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "src/facial_recognition_system/evaluation/metrics.py",
        "configs/e04_arcface.yaml",
        "experiments/results/e01.json",
        "experiments/figures/roc_e01.png",
        "docs/decisions/ADR-001-face-model.md",
        "docs/architecture.png",
        "docs/pipeline.svg",
        "data/README.md",
        "data/splits/lfw_test_pairs.csv",
    ],
)
def test_allowed_files_pass(path: str) -> None:
    assert find_violation(path) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        "data/private/p01/session1/001.jpg",
        "data/private/notes.txt",
        "data/public/lfw/Aaron_Eckhart_0001.jpg",
        "data/gallery.csv",
        "faces/p01.txt",
        "experiments/embeddings/e07.json",
        "checkpoints/e04/best.txt",
        "consent/p01.pdf",
        "notebooks/selfie.JPG",
        "experiments/results/e07_gallery.npy",
        "experiments/results/e07_svm.pkl",
        "model.pt",
        "export/model.onnx",
        "app.sqlite",
        "demo/recording.mp4",
        "src/screenshot.png",
    ],
)
def test_private_or_biometric_files_are_blocked(path: str) -> None:
    assert find_violation(path) is not None


@pytest.mark.unit
def test_windows_separators_are_normalized() -> None:
    assert find_violation("data\\private\\p01\\001.jpg") is not None


@pytest.mark.unit
def test_main_returns_zero_when_all_files_are_allowed(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["README.md", "experiments/figures/roc.png"])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


@pytest.mark.unit
def test_main_reports_every_blocked_file(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["README.md", "faces/p01.txt", "model.pt"])

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "faces/p01.txt" in stderr
    assert "model.pt" in stderr
    assert "README.md:" not in stderr
