from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from natural_scene_classification.constants import CLASS_NAMES


@pytest.fixture()
def tiny_scene_dataset(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    train_root = data_root / "seg_train" / "seg_train"
    test_root = data_root / "seg_test" / "seg_test"
    predict_root = data_root / "seg_pred" / "seg_pred"
    predict_root.mkdir(parents=True, exist_ok=True)

    for root in (train_root, test_root):
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_dir = root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(3):
                pixels = np.full(
                    (32, 32, 3), fill_value=(class_index * 30) + image_index, dtype=np.uint8
                )
                Image.fromarray(pixels).save(class_dir / f"{class_name}_{image_index}.png")

    sample_pixels = np.full((32, 32, 3), fill_value=120, dtype=np.uint8)
    Image.fromarray(sample_pixels).save(predict_root / "sample.png")
    return data_root
