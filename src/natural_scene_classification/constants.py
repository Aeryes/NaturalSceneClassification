from __future__ import annotations

from pathlib import Path

CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_TRAIN_DIR = DEFAULT_DATA_ROOT / "seg_train" / "seg_train"
DEFAULT_TEST_DIR = DEFAULT_DATA_ROOT / "seg_test" / "seg_test"
DEFAULT_PREDICT_DIR = DEFAULT_DATA_ROOT / "seg_pred" / "seg_pred"

DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 3
DEFAULT_MODEL_NAME = "resnet50"
DATASET_SLUG = "puneet6060/intel-image-classification"
