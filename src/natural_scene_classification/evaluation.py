from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from natural_scene_classification.checkpoints import (
    load_model_from_checkpoint,
    save_checkpoint_metadata,
)
from natural_scene_classification.constants import CLASS_NAMES, DEFAULT_TEST_DIR
from natural_scene_classification.data import SceneDataset, enumerate_labeled_items
from natural_scene_classification.training import resolve_device
from natural_scene_classification.transforms import build_eval_transform


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    test_root: Path = DEFAULT_TEST_DIR,
    output_dir: Path | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    device_name: str = "auto",
) -> dict[str, object]:
    device = resolve_device(device_name)
    model, metadata = load_model_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    dataset = SceneDataset(enumerate_labeled_items(test_root), transform=build_eval_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            features = batch_features.to(device)
            labels = batch_labels.to(device)
            outputs = model(features)
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    class_names = metadata.get("class_names", list(CLASS_NAMES))
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred)

    result = {
        "checkpoint_path": str(checkpoint_path),
        "class_names": class_names,
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "per_class": {class_name: report[class_name] for class_name in class_names},
        "confusion_matrix": matrix.tolist(),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        fig, axis = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png")
        plt.close(fig)
        save_checkpoint_metadata(checkpoint_path, output_dir / "checkpoint_metadata.json")

    return result
