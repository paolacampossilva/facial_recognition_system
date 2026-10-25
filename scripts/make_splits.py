"""Builds the identity-disjoint LFW splits, verification pairs and extra unknowns.

Reads the face crops (one subfolder per identity), so photos where MTCNN found
no face are already left out. After splitting, removes the validation and test
copies of photos that also appear in an earlier split (two-stage dHash, decision
D9), and writes the versioned lists and the leakage report to data/splits/.

Usage:
    python scripts/make_splits.py --originals-dir data/public/lfw \
        --crops-dir data/public/lfw_crop160 --output-dir data/splits
"""

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from facial_recognition_system.data import (
    ImagePair,
    LeakageConfig,
    LeakageReport,
    LfwSplits,
    PhotosByIdentity,
    SplitConfig,
    SplitError,
    build_lfw_splits,
    check_leakage,
    exclude_photos,
    hash_photos,
)

CROP_SUFFIX: str = ".png"
PAIR_COLUMNS: tuple[str, ...] = ("path_a", "path_b", "same")
SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")
LEAKAGE_REPORT_FILENAME: str = "lfw_leakage_report.json"


def find_photos(crops_dir: Path) -> PhotosByIdentity:
    """Groups the relative path of every crop by its identity folder."""
    photos: defaultdict[str, list[str]] = defaultdict(list)
    for path in crops_dir.rglob(f"*{CROP_SUFFIX}"):
        relative_path = path.relative_to(crops_dir)
        photos[relative_path.parent.as_posix()].append(relative_path.as_posix())
    return dict(photos)


def write_splits(splits: LfwSplits, output_dir: Path) -> None:
    """Writes the photo lists, pairs and unknowns in the format of the contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    photo_lists = {
        "lfw_train.txt": splits.train,
        "lfw_val.txt": splits.val,
        "lfw_test.txt": splits.test,
        "lfw_val_unknowns.txt": splits.val_unknowns,
        "lfw_test_unknowns.txt": splits.test_unknowns,
    }
    for filename, paths in photo_lists.items():
        (output_dir / filename).write_text("".join(f"{path}\n" for path in paths))

    _write_pairs(splits.val_pairs, output_dir / "lfw_val_pairs.csv")
    _write_pairs(splits.test_pairs, output_dir / "lfw_test_pairs.csv")


def _write_pairs(pairs: Sequence[ImagePair], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(PAIR_COLUMNS)
        writer.writerows((pair.path_a, pair.path_b, int(pair.same)) for pair in pairs)


def _summary(splits: LfwSplits) -> str:
    lines = [f"train: {_identity_count(splits.train)} identities, {len(splits.train)} photos"]
    for name, photos, pairs in (
        ("val", splits.val, splits.val_pairs),
        ("test", splits.test, splits.test_pairs),
    ):
        positives = sum(pair.same for pair in pairs)
        lines.append(
            f"{name}: {_identity_count(photos)} identities, {len(photos)} photos, "
            f"{positives} positive and {len(pairs) - positives} negative pairs"
        )
    lines.append(f"unknowns: {len(splits.val_unknowns)} val, {len(splits.test_unknowns)} test")
    return "\n".join(lines)


def _identity_count(photos: Sequence[str]) -> int:
    return len({path.rsplit("/", 1)[0] for path in photos})


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    defaults = SplitConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--originals-dir", type=Path, default=Path("data/public/lfw"))
    parser.add_argument("--original-suffix", default=".jpg")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/public/lfw_crop160"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--unknowns-per-split", type=int, default=defaults.unknowns_per_split)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Builds and writes the splits, then prints a summary.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.

    Returns:
        0 when the splits are written, 1 when an input folder is missing, the
        dataset cannot be split as configured or a file cannot be read or written.
    """
    args = _parse_args(argv)
    for folder in (args.crops_dir, args.originals_dir):
        if not folder.is_dir():
            print(f"Folder not found: {folder}", file=sys.stderr)
            return 1

    config = SplitConfig(seed=args.seed, unknowns_per_split=args.unknowns_per_split)
    try:
        splits = build_lfw_splits(find_photos(args.crops_dir), config)
        report = _check_leakage(splits, args.originals_dir, args.crops_dir, args.original_suffix)
        splits = exclude_photos(splits, report.excluded)
        write_splits(splits, args.output_dir)
        report_json = report.model_dump_json(indent=2)
        (args.output_dir / LEAKAGE_REPORT_FILENAME).write_text(f"{report_json}\n")
    except SplitError as error:
        print(f"Cannot build the splits: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Failed to read or write files: {error}", file=sys.stderr)
        return 1

    print(_summary(splits))
    print(f"Leakage: {len(report.duplicates)} near-duplicate(s), {len(report.excluded)} removed")
    print(f"Splits written to {args.output_dir}")
    return 0


def _check_leakage(
    splits: LfwSplits, originals_dir: Path, crops_dir: Path, original_suffix: str
) -> LeakageReport:
    config = LeakageConfig()
    paths_by_split = {"train": splits.train, "val": splits.val, "test": splits.test}
    photos = hash_photos(paths_by_split, originals_dir, crops_dir, original_suffix, config)
    return check_leakage(photos, SPLIT_ORDER, config)


if __name__ == "__main__":
    sys.exit(main())
