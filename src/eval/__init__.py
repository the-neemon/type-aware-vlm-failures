"""Evaluation utilities for the type-aware VLM failure project.

Public API:
    is_correct        -- relaxed accuracy judge for a single (gold, prediction) pair
    relaxed_accuracy  -- batch metric over parallel gold/prediction lists
"""

from src.eval.manifest import ManifestRow, read_manifest, write_manifest
from src.eval.relaxed_accuracy import is_correct, relaxed_accuracy

__all__ = [
    "is_correct",
    "relaxed_accuracy",
    "ManifestRow",
    "read_manifest",
    "write_manifest",
]
