"""Train ORMSFD with explicit custom loss and augmentation settings."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'ormsfd.yaml')
    parser.add_argument('--pretrained', default='yolo11n.pt', help="Path or Ultralytics model name; 'none' trains from scratch")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='ormsfd')
    return parser.parse_args()


def remap_detect_pretrained_weights(custom_model, weights_path):
    """Load matching YOLO11 weights, including explicitly remapped neck layers and Detect.

    The historical function name is retained for callers. YAMLs can specify
    pretrained_layer_map: {original_layer: custom_layer}; other layers keep
    their index. Only name-and-shape-compatible tensors are transferred.
    """
    from ultralytics import YOLO

    base_model = YOLO(weights_path)

    source_state = base_model.model.float().state_dict()
    target_state = custom_model.model.state_dict()

    source_detect_idx = len(base_model.model.model) - 1
    target_detect_idx = len(custom_model.model.model) - 1
    layer_map = {int(src): int(dst) for src, dst in
                 custom_model.model.yaml.get('pretrained_layer_map', {}).items()}
    layer_map[source_detect_idx] = target_detect_idx
    for src, dst in layer_map.items():
        if not (0 <= src < len(base_model.model.model) and 0 <= dst < len(custom_model.model.model)):
            raise ValueError(f'Invalid pretrained layer mapping: {src} -> {dst}')
        if type(base_model.model.model[src]) is not type(custom_model.model.model[dst]):
            raise ValueError(f'Pretrained layer types differ at {src} -> {dst}')

    remapped = {}
    detect_count = 0
    neck_counts = {}
    for source_key, source_value in source_state.items():
        if not source_key.startswith('model.'):
            continue
        _, source_idx, suffix = source_key.split('.', 2)
        source_idx = int(source_idx)
        target_idx = layer_map.get(source_idx, source_idx)
        target_key = f'model.{target_idx}.{suffix}'
        if target_key in target_state and source_value.shape == target_state[target_key].shape:
            remapped[target_key] = source_value
            if source_idx == source_detect_idx:
                detect_count += 1
            elif source_idx in layer_map:
                neck_counts[(source_idx, target_idx)] = neck_counts.get((source_idx, target_idx), 0) + 1

    custom_model.model.load_state_dict(remapped, strict=False)
    # Model.train() passes this initialized network to the trainer only when
    # ckpt is populated, as Model.load() normally does. Retain that behavior
    # without a preliminary same-index load that could contaminate new blocks.
    custom_model.ckpt = base_model.ckpt
    custom_model.overrides['pretrained'] = str(weights_path)
    print(
        f"Detect remap: model.{source_detect_idx} -> model.{target_detect_idx}, "
        f"transferred {detect_count} items"
    )
    for (src, dst), count in sorted(neck_counts.items()):
        print(f'Neck remap: model.{src} -> model.{dst}, transferred {count} items')
    print(
        f"Effective original pretrained items: "
        f"{len(remapped)}/{len(target_state)}"
    )

    del base_model
    return custom_model


def main():
    args = parse_args()
    for path in [args.data, args.model]:
        if not path.is_file():
            raise FileNotFoundError(path)
    settings = {
        'BBOX_LOSS_TYPE': 'mpdiou',
        'MPDIOU_WEIGHT': '1.0',
        'USE_REPLOSS': '1',
        'REP_ALPHA': '0.0001', 'REP_BETA': '0.0',
        'REP_PNMS': '0.45', 'REP_GTNMS': '0.4', 'REP_MARGIN': '0.05',
        'TAL_TOPK': '10', 'TAL_TOPK2': '10', 'CLS_POS_WEIGHT': '1.0',
    }
    os.environ.update(settings)
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(args.seed, deterministic=True)
    model = YOLO(str(args.model.resolve()))
    pretrained = args.pretrained.lower() != 'none'
    if pretrained:
        remap_detect_pretrained_weights(model, args.pretrained)

    def save_custom_settings(trainer):
        output = Path(trainer.save_dir) / 'ormsfd_settings.json'
        metadata = dict(settings, overlap_prior=model.model.yaml.get('overlap_prior', {}),
                        pretrained_layer_map=model.model.yaml.get('pretrained_layer_map', {}))
        output.write_text(json.dumps(metadata, indent=2) + '\n')

    model.add_callback('on_train_start', save_custom_settings)
    model.train(
        data=str(args.data.resolve()), seed=args.seed, deterministic=True,
        pretrained=args.pretrained if pretrained else False,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        device=args.device, project=str(Path(args.project).resolve()), name=args.name,
        optimizer='AdamW', lr0=0.001, lrf=0.01, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=True, mosaic=1.0, mixup=0.1, close_mosaic=5,
        degrees=5.0, translate=0.1, scale=0.50, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,
    )
    print(f'Best weights: {model.trainer.best}')


if __name__ == '__main__':
    main()
