"""Audits the versioned LFW splits for photos shared across splits (section 30, decision D9).

Hashes every photo listed in data/splits/lfw_{train,val,test}.txt with the
two-stage dHash and fails when a near-duplicate crosses splits. The splits from
scripts/make_splits.py already have those copies removed, so this should pass.

Usage:
    python scripts/check_leakage.py --originals-dir data/public/lfw \
        --crops-dir data/public/lfw_crop160 --splits-dir data/splits
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from facial_recognition_system.data import LeakageConfig, check_leakage, hash_photos

SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--originals-dir", type=Path, default=Path("data/public/lfw"))
    parser.add_argument("--original-suffix", default=".jpg")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/public/lfw_crop160"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Checks the split lists and prints every near-duplicate found.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.

    Returns:
        0 when no photo crosses splits, 1 when one does or a file cannot be read.
    """
    args = _parse_args(argv)
    config = LeakageConfig()
    try:
        paths_by_split = {
            split: (args.splits_dir / f"lfw_{split}.txt").read_text().split()
            for split in SPLIT_ORDER
        }
        photos = hash_photos(
            paths_by_split, args.originals_dir, args.crops_dir, args.original_suffix, config
        )
    except OSError as error:
        print(f"Failed to read the splits or photos: {error}", file=sys.stderr)
        return 1

    report = check_leakage(photos, SPLIT_ORDER, config)
    for duplicate in report.duplicates:
        print(
            f"{duplicate.split_a}:{duplicate.path_a} = {duplicate.split_b}:{duplicate.path_b} "
            f"(distances {duplicate.candidate_distance} and {duplicate.confirm_distance})",
            file=sys.stderr,
        )

    checked = ", ".join(f"{count} {split}" for split, count in report.photos_per_split.items())
    if report.duplicates:
        print(f"Leakage: {len(report.duplicates)} near-duplicate(s) across splits", file=sys.stderr)
        return 1

    print(f"No near-duplicates across splits ({checked} photos checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
