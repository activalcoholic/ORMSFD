# ORMSFD

ORMSFD is an underwater fish detector for Low activity, Routine swimming, and Feeding aggregation scenarios. The model is based on YOLO11 and uses MSFEM with a shared image-predicted overlap prior, sequential feature fusion, MPDIoU, and RepGT.

## Installation

Python 3.10 is recommended. First install PyTorch and torchvision suitable for your device, then run:

```bash
git clone https://github.com/activalcoholic/ORMSFD.git
cd ORMSFD
python -m pip install -e .
```


## Dataset

Download the **Underwater Fish Dataset**: [GitHub Release](https://github.com/activalcoholic/ORMSFD/releases/tag/underwater-fish-dataset).

| File | Contents |
| --- | --- |
| [Dataset-train.zip](https://github.com/activalcoholic/ORMSFD/releases/download/underwater-fish-dataset/Underwater-Fish-Dataset-train.zip) | Training set: 3980 images and their labels |
| [Dataset-val.zip](https://github.com/activalcoholic/ORMSFD/releases/download/underwater-fish-dataset/Underwater-Fish-Dataset-val.zip) | Validation set: 740 images and their labels |
| [Dataset-test.zip](https://github.com/activalcoholic/ORMSFD/releases/download/underwater-fish-dataset/Underwater-Fish-Dataset-test.zip) | Test set: three scenarios, each with 200 images and their labels |

Extract all three archives into the repository's `data/` directory. Each scenario folder in the test set contains `images` and `labels`:

```text
data/Underwater Fish Dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── test/
│   ├── Low activity/
│   │   ├── images/
│   │   └── labels/
│   ├── Routine swimming/
│   │   ├── images/
│   │   └── labels/
│   └── Feeding aggregation/
│       ├── images/
│       └── labels/
├── data.yaml
├── data_low_activity.yaml
├── data_routine_swimming.yaml
└── data_feeding_aggregation.yaml
```

Labels use the YOLO format. The four dataset configuration files in the repository root can be used directly; the archives also include configurations with paths relative to the dataset directory.

## Training

```bash
python train.py --data data.yaml --device 0
```

The default model configuration is [`ormsfd.yaml`](ormsfd.yaml). The default training settings are 100 epochs, an image size of 640, a batch size of 32, the AdamW optimizer, an initial learning rate of 0.001, and a random seed of 32.

```bash
# Reduce the batch size if GPU memory is insufficient
python train.py --data data.yaml --device 0 --batch 8
```

The default initialization uses `yolo11n.pt`. Training maps compatible tensors to the revised neck layers using `pretrained_layer_map` and maps Detect weights to the final layer. Use `--pretrained none` to train from scratch.

A single P3 head predicts overlap-prior logits. MSFEM pools the predicted probabilities for the other feature scales and fuses the prior with half of its directional-feature channels, while preserving the other half at this fusion step. Enhanced features feed both detection and the subsequent bottom-up fusion stage. Both training and inference use predicted priors.

Training outputs are saved in `runs/detect/`, including the best weights, actual training arguments, and custom loss and prior settings in `ormsfd_settings.json`.

## Evaluation and Inference

Replace `--weights` with the path to the `best.pt` file generated during training:

```bash
# Full test set
python evaluate.py --weights path/to/best.pt --data data.yaml

# Evaluate the three scenarios separately
python evaluate.py --weights path/to/best.pt --data data_low_activity.yaml --name low_activity
python evaluate.py --weights path/to/best.pt --data data_routine_swimming.yaml --name routine_swimming
python evaluate.py --weights path/to/best.pt --data data_feeding_aggregation.yaml --name feeding_aggregation

# Image or video inference
python predict.py --weights path/to/best.pt --source path/to/image.jpg --device 0
```

Evaluation reports Precision, Recall, F1, mAP@0.5, and mAP@0.5:0.95. Evaluation plots are saved under `runs/evaluate/`.

## Code Locations

| Function | File |
| --- | --- |
| MSFEM and channel-split prior fusion | `ultralytics/nn/modules/MSFEM.py` |
| Shared overlap-prior head | `ultralytics/nn/modules/overlap_prior.py` |
| Overlap targets and auxiliary supervision | `ultralytics/utils/overlap_prior.py`, `ultralytics/utils/loss.py` |
| Module imports and model parsing | `ultralytics/nn/modules/__init__.py`, `ultralytics/nn/tasks.py` |
| MPDIoU, RepGT | `ultralytics/utils/loss.py` |
| Model configuration | `ormsfd.yaml` |

## License

The code retains the AGPL-3.0 license of the original Ultralytics project. See [`LICENSE`](LICENSE).

