"""Public dataset splits and verification pairs."""

from facial_recognition_system.data.errors import (
    NotEnoughNegativePairsError,
    NotEnoughUnknownsError,
    SplitError,
    UnknownSplitError,
)
from facial_recognition_system.data.leakage import (
    HashedPhoto,
    LeakageConfig,
    LeakageReport,
    NearDuplicate,
    check_leakage,
    dhash,
    find_cross_split_duplicates,
    hamming_distance,
    hash_photos,
)
from facial_recognition_system.data.splits import (
    ImagePair,
    LfwSplits,
    PhotosByIdentity,
    SplitConfig,
    build_lfw_splits,
    exclude_photos,
)

__all__ = [
    "HashedPhoto",
    "ImagePair",
    "LeakageConfig",
    "LeakageReport",
    "LfwSplits",
    "NearDuplicate",
    "NotEnoughNegativePairsError",
    "NotEnoughUnknownsError",
    "PhotosByIdentity",
    "SplitConfig",
    "SplitError",
    "UnknownSplitError",
    "build_lfw_splits",
    "check_leakage",
    "dhash",
    "exclude_photos",
    "find_cross_split_duplicates",
    "hamming_distance",
    "hash_photos",
]
