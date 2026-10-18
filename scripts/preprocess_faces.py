"""Detects, crops and saves one face per image for a whole dataset folder.

The input folder holds one subfolder per identity (LFW layout, also used for
data/private/). The output mirrors that layout with PNG face crops, plus a
JSON report with the configuration and every image that failed.

Usage:
    python scripts/preprocess_faces.py --input-dir data/public/lfw \
        --output-dir data/public/lfw_crop160 --selection center
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import BaseModel
from tqdm import tqdm

from facial_recognition_system.preprocessing import (
    AlignmentMode,
    FaceDetector,
    FacePreprocessor,
    FaceSelection,
    MTCNNDetector,
    PreprocessingConfig,
    PreprocessingError,
    load_image,
)

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
REPORT_FILENAME: str = "preprocessing_report.json"


class FailedImage(BaseModel):
    path: str
    reason: str


class PreprocessingReport(BaseModel):
    """Summary of one run, saved next to the crops for reproducibility."""

    input_dir: str
    config: PreprocessingConfig
    total: int
    processed: int
    skipped: int
    failures: list[FailedImage]


def find_images(input_dir: Path) -> list[Path]:
    """Lists every image under ``input_dir`` in a stable order."""
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    preprocessor: FacePreprocessor,
    config: PreprocessingConfig,
    force: bool = False,
) -> PreprocessingReport:
    """Saves a face crop for every image and records the ones that failed.

    Args:
        input_dir: Dataset folder with one subfolder per identity.
        output_dir: Destination; crops keep the relative path with a ``.png`` suffix.
        preprocessor: Pipeline that turns an image into a face crop.
        config: Configuration of ``preprocessor``, stored in the report.
        force: Reprocesses images whose crop already exists.

    Returns:
        The report, also written to ``output_dir / REPORT_FILENAME``.
    """
    images = find_images(input_dir)
    failures: list[FailedImage] = []
    processed = 0
    skipped = 0

    for image_path in tqdm(images, desc="faces", unit="img"):
        relative_path = image_path.relative_to(input_dir)
        output_path = (output_dir / relative_path).with_suffix(".png")
        if output_path.exists() and not force:
            skipped += 1
            continue

        try:
            face = preprocessor.process(load_image(image_path))
        except PreprocessingError as error:
            failures.append(FailedImage(path=relative_path.as_posix(), reason=str(error)))
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        face.save(output_path)
        processed += 1

    report = PreprocessingReport(
        input_dir=input_dir.as_posix(),
        config=config,
        total=len(images),
        processed=processed,
        skipped=skipped,
        failures=failures,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_FILENAME).write_text(report.model_dump_json(indent=2))
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = PreprocessingConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=FaceSelection,
        choices=list(FaceSelection),
        default=defaults.selection,
    )
    parser.add_argument(
        "--alignment",
        type=AlignmentMode,
        choices=list(AlignmentMode),
        default=defaults.alignment,
    )
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--margin", type=int, default=defaults.margin)
    parser.add_argument("--min-probability", type=float, default=defaults.min_probability)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="reprocess existing crops")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, detector: FaceDetector | None = None) -> int:
    """Preprocesses a dataset folder and prints a summary.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.
        detector: Face detector to use (injected in tests). Defaults to MTCNN.

    Returns:
        0 when the run completes (failed images are listed in the report),
        1 when the input folder is missing or the output cannot be written.
    """
    args = _parse_args(argv)
    if not args.input_dir.is_dir():
        print(f"Input folder not found: {args.input_dir}", file=sys.stderr)
        return 1

    config = PreprocessingConfig(
        image_size=args.image_size,
        margin=args.margin,
        min_probability=args.min_probability,
        selection=args.selection,
        alignment=args.alignment,
    )
    preprocessor = FacePreprocessor(detector or MTCNNDetector(device=args.device), config)

    try:
        report = preprocess_directory(
            args.input_dir, args.output_dir, preprocessor, config, force=args.force
        )
    except OSError as error:
        print(f"Failed to write face crops: {error}", file=sys.stderr)
        return 1

    print(
        f"{report.processed} processed, {report.skipped} skipped, "
        f"{len(report.failures)} failed out of {report.total} images. "
        f"Report: {args.output_dir / REPORT_FILENAME}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
