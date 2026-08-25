from __future__ import annotations

from pathlib import Path

from natural_scene_classification.constants import CLASS_NAMES
from natural_scene_classification.data import (
    create_split_manifest,
    inspect_dataset,
    read_split_manifest,
)

IMAGES_PER_CLASS = 3


def test_inspect_dataset_counts_classes(tiny_scene_dataset: Path) -> None:
    summary = inspect_dataset(tiny_scene_dataset / "seg_train" / "seg_train")
    assert summary["total_images"] == len(CLASS_NAMES) * IMAGES_PER_CLASS
    assert summary["class_counts"]["buildings"] == IMAGES_PER_CLASS


def test_split_manifest_is_deterministic(tiny_scene_dataset: Path, tmp_path: Path) -> None:
    manifest_one = create_split_manifest(
        train_root=tiny_scene_dataset / "seg_train" / "seg_train",
        output_path=tmp_path / "one.csv",
        seed=42,
    )
    manifest_two = create_split_manifest(
        train_root=tiny_scene_dataset / "seg_train" / "seg_train",
        output_path=tmp_path / "two.csv",
        seed=42,
    )
    assert manifest_one.read_text(encoding="utf-8") == manifest_two.read_text(encoding="utf-8")


def test_read_split_manifest_returns_train_and_validation(
    tiny_scene_dataset: Path, tmp_path: Path
) -> None:
    manifest = create_split_manifest(
        train_root=tiny_scene_dataset / "seg_train" / "seg_train",
        output_path=tmp_path / "split.csv",
        seed=42,
    )
    splits = read_split_manifest(manifest, tiny_scene_dataset / "seg_train" / "seg_train")
    assert splits["train"]
    assert splits["validation"]
