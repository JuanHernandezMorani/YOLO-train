"""YOLOv11 dual-head architecture for pixel-art segmentation and PBR map classification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    if name.lower() == "silu":
        return nn.SiLU(inplace=True)
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unsupported activation '{name}'.")


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + activation block used throughout the network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True, activation: str = "silu") -> None:
        super().__init__()
        hidden_channels = channels // 2
        self.conv1 = ConvBNAct(channels, hidden_channels, 1, activation=activation)
        self.conv2 = ConvBNAct(hidden_channels, channels, 3, activation=activation)
        self.use_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        if self.use_shortcut:
            y = y + x
        return y


class C3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden_channels, 1, activation=activation)
        self.cv2 = ConvBNAct(in_channels, hidden_channels, 1, activation=activation)
        self.cv3 = ConvBNAct(2 * hidden_channels, out_channels, 1, activation=activation)
        self.m = nn.Sequential(
            *[
                Bottleneck(hidden_channels, shortcut, activation=activation)
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling - fast variant."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden_channels, 1, activation=activation)
        self.cv2 = ConvBNAct(hidden_channels * 4, out_channels, 1, activation=activation)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPDarknetBackbone(nn.Module):
    def __init__(self, activation: str = "silu") -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 64, 3, 2, activation=activation)
        self.stage1 = nn.Sequential(
            ConvBNAct(64, 128, 3, 2, activation=activation),
            C3(128, 128, n=1, activation=activation),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(128, 256, 3, 2, activation=activation),
            C3(256, 256, n=3, activation=activation),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(256, 512, 3, 2, activation=activation),
            C3(512, 512, n=1, activation=activation),
        )
        self.sppf = SPPF(512, 512, activation=activation)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        c5 = self.sppf(c5)
        return [c3, c4, c5]


class PANFPN(nn.Module):
    def __init__(self, activation: str = "silu") -> None:
        super().__init__()
        self.lateral5 = ConvBNAct(512, 256, 1, activation=activation)
        self.lateral4 = ConvBNAct(256, 128, 1, activation=activation)
        self.lateral3 = ConvBNAct(128, 128, 1, activation=activation)

        self.reduce4 = ConvBNAct(256 + 256, 256, 1, activation=activation)
        self.reduce3 = ConvBNAct(128 + 128, 128, 1, activation=activation)

        self.down4 = ConvBNAct(128, 256, 3, 2, activation=activation)
        self.down5 = ConvBNAct(256, 512, 3, 2, activation=activation)

        self.output4 = C3(256 + 256, 256, n=1, activation=activation)
        self.output5 = C3(512 + 512, 512, n=1, activation=activation)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        c3, c4, c5 = features

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3)

        up4 = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4 = self.reduce4(torch.cat([p4, up4], dim=1))

        up3 = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = self.reduce3(torch.cat([p3, up3], dim=1))

        n4 = self.output4(torch.cat([self.down4(p3), p4], dim=1))
        n5 = self.output5(torch.cat([self.down5(n4), p5], dim=1))

        return [p3, n4, n5]


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.proj_layers = nn.ModuleList(
            [ConvBNAct(c, 128, 3, activation=activation) for c in in_channels]
        )
        self.final_conv = nn.Conv2d(128 * len(in_channels), num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = features[0].shape[-2:]
        upsampled: List[torch.Tensor] = []
        for feat, proj in zip(features, self.proj_layers):
            x = proj(feat)
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            upsampled.append(x)
        fused = torch.cat(upsampled, dim=1)
        return self.final_conv(fused)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pooling: str = "avg",
    ) -> None:
        super().__init__()
        if pooling not in {"avg", "max"}:
            raise ValueError("pooling must be 'avg' or 'max'")
        self.pooling = pooling
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        if self.pooling == "avg":
            x = F.adaptive_avg_pool2d(x, 1)
        else:
            x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@dataclass
class ModelOutput:
    segmentation: torch.Tensor
    pbr_logits: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {"segmentation": self.segmentation, "pbr_logits": self.pbr_logits}


class YOLOv11DualHead(nn.Module):
    """YOLOv11 model with shared backbone and PAN-FPN for segmentation + PBR classification."""

    def __init__(
        self,
        seg_classes: int,
        pbr_classes: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.activation = activation
        self.backbone = CSPDarknetBackbone(activation=activation)
        self.neck = PANFPN(activation=activation)
        self.seg_head = SegmentationHead([128, 256, 512], seg_classes, activation=activation)
        self.pbr_head = ClassificationHead(512, pbr_classes, pooling="avg")

    def forward(self, image: torch.Tensor, pbr_map: torch.Tensor) -> ModelOutput:
        image_features = self.neck(self.backbone(image))
        seg_logits = self.seg_head(image_features)

        pbr_features = self.neck(self.backbone(pbr_map))
        pbr_logits = self.pbr_head(pbr_features)

        return ModelOutput(segmentation=seg_logits, pbr_logits=pbr_logits)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, pbr_map: torch.Tensor) -> ModelOutput:
        self.eval()
        return self.forward(image, pbr_map)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "YOLOv11DualHead":
        import yaml

        cfg_path = Path(config_path)
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        model_cfg = cfg.get("model", {})
        heads_cfg = model_cfg.get("heads", {})
        backbone_cfg = model_cfg.get("backbone", {})

        seg_classes = heads_cfg.get("seg_head", {}).get("num_classes", 18)
        pbr_classes = heads_cfg.get("pbr_head", {}).get("num_classes", 16)
        activation = backbone_cfg.get("activation", "silu")

        return cls(seg_classes=seg_classes, pbr_classes=pbr_classes, activation=activation)


__all__ = [
    "YOLOv11DualHead",
    "ModelOutput",
    "SegmentationHead",
    "ClassificationHead",
]
