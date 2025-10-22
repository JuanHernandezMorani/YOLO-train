"""Data handling utilities for the YOLOv11 dual-head training pipeline."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PBR_CLASS_NAMES: Sequence[str] = (
    "ao",
    "curvature",
    "emissive",
    "fuzz",
    "height",
    "ior",
    "material",
    "metallic",
    "normal",
    "opacity",
    "porosity",
    "roughness",
    "specular",
    "structural",
    "subsurface",
    "transmission",
)


@dataclass
class Sample:
    image_path: Path
    label_path: Optional[Path]
    pbr_paths: List[Path]


class DualHeadPixelPBRDataset(Dataset):
    """Dataset that pairs base pixel-art images with segmentation masks and PBR maps."""

    def __init__(
        self,
        root: Path,
        image_dir: Path,
        label_dir: Path,
        pbr_dir: Path,
        seg_classes: int,
        pbr_class_names: Sequence[str] = PBR_CLASS_NAMES,
        image_size: int = 640,
        augment: bool = False,
        random_pbr: bool = True,
        keep_missing: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.pbr_dir = Path(pbr_dir)
        self.seg_classes = seg_classes
        self.pbr_class_names = tuple(pbr_class_names)
        self.pbr_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.pbr_class_names)
        }
        self.image_size = image_size
        self.augment = augment
        self.random_pbr = random_pbr
        self.keep_missing = keep_missing

        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = transforms.Resize(
            (image_size, image_size), interpolation=Image.NEAREST
        )

        self.samples: List[Sample] = []
        self._build_index()

    def _build_index(self) -> None:
        image_files = sorted(
            [
                *self.image_dir.glob("*.png"),
                *self.image_dir.glob("*.jpg"),
                *self.image_dir.glob("*.jpeg"),
                *self.image_dir.glob("*.bmp"),
            ]
        )
        for image_path in image_files:
            base = image_path.stem
            label_path = self._find_label(base)
            pbr_paths = sorted(self.pbr_dir.glob(f"{base}_*.png"))
            if not pbr_paths and not self.keep_missing:
                continue
            self.samples.append(Sample(image_path=image_path, label_path=label_path, pbr_paths=pbr_paths))

    def _find_label(self, base: str) -> Optional[Path]:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            candidate = self.label_dir / f"{base}{ext}"
            if candidate.exists():
                return candidate
        candidate = self.label_dir / f"{base}.txt"
        if candidate.exists():
            return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        tensor = self.base_transform(image)
        if self.augment:
            tensor = self._apply_default_augmentations(tensor)
        return tensor

    def _load_mask(self, path: Optional[Path], size: Tuple[int, int]) -> torch.Tensor:
        if path is None or not path.exists():
            return torch.zeros(size, dtype=torch.long)
        if path.suffix.lower() == ".txt":
            return self._mask_from_yolo(path, size)
        mask_image = Image.open(path)
        if mask_image.mode not in ("L", "P"):
            mask_image = mask_image.convert("L")
        mask_image = self.mask_transform(mask_image)
        mask = torch.from_numpy(np.array(mask_image, dtype=np.int64))
        return mask

    def _mask_from_yolo(self, path: Path, size: Tuple[int, int]) -> torch.Tensor:
        mask = torch.zeros(size, dtype=torch.long)

        try:
            with path.open("r", encoding="utf-8") as handle:
                annotations = [line.strip().split() for line in handle if line.strip()]
        except (FileNotFoundError, UnicodeDecodeError) as exc:
            print(f"Warning: Could not read label file {path}: {exc}")
            return mask

        if not annotations:
            return mask

        width, height = size[1], size[0]
        mask_image = Image.new("L", (width, height), 0)
        drawer = ImageDraw.Draw(mask_image)

        valid_annotations = 0
        for idx, anno in enumerate(annotations):
            try:
                if len(anno) < 6:
                    continue

                cls = int(float(anno[0]))
                coords = [float(v) for v in anno[1:]]

                if len(coords) % 2 != 0 or len(coords) < 6:
                    continue

                points = []
                for i in range(0, len(coords), 2):
                    x = min(width - 1, max(0, int(round(coords[i] * width))))
                    y = min(height - 1, max(0, int(round(coords[i + 1] * height))))
                    points.append((x, y))

                if len(points) < 3 or len(set(points)) < 3:
                    continue

                if points[0] != points[-1]:
                    points.append(points[0])

                drawer.polygon(points, outline=cls, fill=cls)
                valid_annotations += 1

            except (ValueError, IndexError) as exc:
                print(f"Warning: Invalid annotation in {path}, line {idx}: {exc}")
                continue

        if valid_annotations < len(annotations):
            print(f"Warning: {len(annotations) - valid_annotations} invalid annotations in {path}")

        mask = torch.from_numpy(np.array(mask_image, dtype=np.int64))
        return mask

    def _load_pbr_map(self, paths: List[Path]) -> Tuple[torch.Tensor, int, str]:
        if not paths:
            blank = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return blank, -1, "unknown"
        selected = random.choice(paths) if self.random_pbr else paths[0]
        image = Image.open(selected)
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.base_transform(image)
        suffix = selected.stem.split("_")[-1]
        label = self.pbr_to_index.get(suffix, -1)
        return tensor, label, suffix

    def _apply_default_augmentations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Augmentaciones específicas para pixel-art que preservan bordes nítidos."""
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])

        if random.random() < 0.3:
            tensor = torch.flip(tensor, dims=[1])

        if random.random() < 0.1:
            k = random.randint(1, 3)
            tensor = torch.rot90(tensor, k, dims=[1, 2])

        if random.random() < 0.2:
            brightness = random.uniform(0.95, 1.05)
            contrast = random.uniform(0.95, 1.05)
            tensor = transforms.functional.adjust_brightness(tensor, brightness)
            tensor = transforms.functional.adjust_contrast(tensor, contrast)

        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        mask = self._load_mask(sample.label_path, size=(self.image_size, self.image_size))
        pbr_map, pbr_label, pbr_name = self._load_pbr_map(sample.pbr_paths)
        return {
            "image": image,
            "mask": mask,
            "pbr_map": pbr_map,
            "pbr_label": torch.tensor(pbr_label, dtype=torch.long),
            "meta": {
                "image_path": str(sample.image_path),
                "pbr_name": pbr_name,
            },
        }


def dualhead_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    pbr_maps = torch.stack([item["pbr_map"] for item in batch], dim=0)
    pbr_labels = torch.stack([item["pbr_label"] for item in batch], dim=0)
    metas = [item["meta"] for item in batch]
    return {
        "images": images,
        "masks": masks,
        "pbr_maps": pbr_maps,
        "pbr_labels": pbr_labels,
        "meta": metas,
    }


def create_dataloader(
    dataset: DualHeadPixelPBRDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dualhead_collate_fn,
    )


def export_class_mappings(path: Path, dataset: DualHeadPixelPBRDataset) -> None:
    payload = {
        "segmentation_classes": dataset.seg_classes,
        "pbr_class_names": list(dataset.pbr_class_names),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


__all__ = [
    "DualHeadPixelPBRDataset",
    "dualhead_collate_fn",
    "create_dataloader",
    "export_class_mappings",
    "PBR_CLASS_NAMES",
]
