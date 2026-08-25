# Model Card: Natural Scene Classification

## Summary
This repository trains a ResNet50 transfer-learning classifier for six outdoor scene classes from the Kaggle Intel Image Classification dataset:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

## Intended use
- Educational computer-vision experiments
- Reproducible training and evaluation workflows
- Portfolio demonstration of dataset handling, transfer learning, and offline inference

## Out of scope
- Safety-critical decision making
- Domain transfer beyond the source dataset without re-validation
- Claims of generalized performance outside the documented dataset split

## Architecture
- Backbone: torchvision `resnet50`
- Transfer-learning head:
  - `Linear(in_features, 512)`
  - `ReLU`
  - `Dropout(0.1)`
  - `Linear(512, 256)`
  - `ReLU`
  - `Dropout(0.1)`
  - `Linear(256, 6)`

## Data
- Source: Kaggle dataset `puneet6060/intel-image-classification`
- Raw held-out split: `data/seg_test/seg_test`
- Training source for deterministic train/validation split: `data/seg_train/seg_train`
- Local dataset metadata and checksums are stored in `configs/intel-scenes.dataset.json` after `nsc data download` or `nsc data inspect`

## Preprocessing
- Train: resize, horizontal flip, rotation, resized crop, color jitter, ImageNet normalization
- Eval: resize and ImageNet normalization

## Metrics
- Historical repository README claim: `92.20%` accuracy
- Verified metrics from modernized runs are written to `artifacts/runs/<run-id>/metrics/test_metrics.json`
- Verified baseline run: `artifacts/runs/20260825-200645/metrics/test_metrics.json`
  - held-out `seg_test` accuracy: `92.37%`
  - macro precision / recall / F1: `92.50%` / `92.59%` / `92.50%`
  - weighted precision / recall / F1: `92.39%` / `92.37%` / `92.33%`
  - best checkpoint SHA-256: `cc1584d47d20916014cde6cf8b2efa81e7102b8cbde11a0a00ec75dbfda3d698`
- Do not treat historical and modernized results as directly comparable when the split protocol changes

## Reproducibility
- Default seed: `42`
- Best and last checkpoints include class names, config, epoch, metrics, and training history
- Checkpoint metadata includes SHA-256 hashes

## Limitations
- Performance depends on the exact Kaggle download contents and deterministic split seed
- The model is trained on a narrow benchmark and may fail under distribution shift
- Confidence scores are not calibrated probabilities for arbitrary downstream use
