"""Inference utility for the YOLOv11 dual-head model."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.yolov11_dualhead import YOLOv11DualHead
from utils.data import PBR_CLASS_NAMES


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the YOLOv11 dual-head model")
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image path")
    parser.add_argument("--pbr-map", type=Path, required=True, help="Associated PBR map image path")
    parser.add_argument("--weights", type=Path, default=Path("runs/dualhead_pixel_pbr/checkpoints/best.pt"), help="Model checkpoint")
    parser.add_argument("--data", type=Path, default=Path("dataset.yaml"), help="Dataset configuration for class mappings")
    parser.add_argument("--model-config", type=Path, default=Path("yolov11_dualhead.yaml"), help="Model configuration file")
    parser.add_argument("--device", type=str, default=None, help="Torch device override")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference size")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--mask-output", type=Path, default=None, help="Optional PNG mask output path")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(imgsz: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
        ]
    )


def load_checkpoint(weights: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(weights, map_location=device)
    candidates = ("model_state_dict", "model", "state_dict")
    if isinstance(checkpoint, dict):
        for key in candidates:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    raise ValueError(f"Unsupported checkpoint format from {weights}")


def save_mask(mask: torch.Tensor, path: Path, palette: np.ndarray | None = None) -> str:
    mask = mask.cpu().numpy().astype(np.uint8)
    image = Image.fromarray(mask, mode="P")
    if palette is not None:
        palette = palette.flatten().tolist()
        image.putpalette(palette)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = select_device(args.device)

    data_cfg = load_yaml(args.data)
    model_cfg = load_yaml(args.model_config)
    seg_cfg = data_cfg.get("segmentation", {})
    pbr_cfg = data_cfg.get("pbr_fusion", {})
    seg_classes = seg_cfg.get("num_classes", 18)
    pbr_classes = pbr_cfg.get("num_classes", 16)
    pbr_names = pbr_cfg.get("class_names", list(PBR_CLASS_NAMES))

    imgsz = args.imgsz or model_cfg.get("training", {}).get("imgsz", 640)
    transform = build_transforms(imgsz)

    model = YOLOv11DualHead(seg_classes=seg_classes, pbr_classes=pbr_classes)
    checkpoint = load_checkpoint(args.weights, device)
    if any("pbr_head" in key for key in checkpoint):
        model.load_state_dict(checkpoint, strict=False)
        LOGGER.info("Loaded full dual-head checkpoint including PBR head from %s", args.weights)
    else:
        LOGGER.info("Checkpoint missing PBR head weights. Loading backbone/neck/segmentation head from %s", args.weights)
        model.load_pretrained_weights(checkpoint, strict=False)
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    pbr_image = Image.open(args.pbr_map).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    pbr_tensor = transform(pbr_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.predict(image_tensor, pbr_tensor)
        segmentation_logits = outputs.segmentation
        pbr_logits = outputs.pbr_logits

    segmentation_mask = segmentation_logits.argmax(dim=1)[0]
    pbr_probs = torch.softmax(pbr_logits, dim=1)
    confidence, pred_idx = torch.max(pbr_probs, dim=1)
    pred_idx = pred_idx.item()
    confidence = confidence.item()
    pbr_label = pbr_names[pred_idx] if 0 <= pred_idx < len(pbr_names) else "unknown"

    if args.mask_output is not None:
        mask_path = args.mask_output
    else:
        mask_name = args.image.stem + "_mask.png"
        mask_path = Path("runs/dualhead_pixel_pbr/inference") / mask_name
    palette = np.linspace(0, 255, num=seg_classes, dtype=np.uint8)
    palette = np.stack([palette, np.roll(palette, 85), np.roll(palette, 170)], axis=1)
    mask_path_str = save_mask(segmentation_mask, mask_path, palette=palette)

    result = {
        "image": str(args.image),
        "pbr_map": str(args.pbr_map),
        "segmentation_mask": mask_path_str,
        "pbr_prediction": {
            "label": pbr_label,
            "confidence": confidence,
        },
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
