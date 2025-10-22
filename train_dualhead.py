"""Training script for the YOLOv11 dual-head pixel-art + PBR fusion model."""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter

try:
    import GPUtil
except ImportError:  # pragma: no cover - optional dependency in some environments
    GPUtil = None

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency in some environments
    psutil = None

from collections import deque
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from models.yolov11_dualhead import YOLOv11DualHead
from utils.data import (
    DualHeadPixelPBRDataset,
    PBR_CLASS_NAMES,
    create_dataloader,
    export_class_mappings,
)


LOGGER = logging.getLogger(__name__)


class LiveLogHandler(logging.Handler):
    """Collect log records for live console rendering."""

    def __init__(self, buffer: deque[str], lock: threading.Lock, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.buffer = buffer
        self.lock = lock

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - interactive output
        msg = self.format(record)
        with self.lock:
            self.buffer.append(msg)


def _format_uptime(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ResourceMonitor:
    """Gather and render real-time system metrics for the live dashboard."""

    def __init__(self, start_time: float) -> None:
        self.start_time = start_time
        self._lock = threading.Lock()
        self._last_disk = psutil.disk_io_counters() if psutil else None
        self._last_time = time.time()
        if psutil:
            psutil.cpu_percent(interval=None)  # Prime internal state
        self._amdgpu_available = shutil.which("amdgpu_top") is not None

    def _gpu_metrics(self) -> dict[str, str]:  # pragma: no cover - hardware dependent
        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
            except Exception:
                gpus = []
            if gpus:
                gpu = gpus[0]
                load = float(gpu.load or 0.0) * 100
                memory_total = float(getattr(gpu, "memoryTotal", 0.0))
                memory_used = float(getattr(gpu, "memoryUsed", 0.0))
                temperature = getattr(gpu, "temperature", None)
                return {
                    "load": load,
                    "rest": max(0.0, 100.0 - load),
                    "memory_used": memory_used,
                    "memory_total": memory_total,
                    "temperature": temperature,
                    "name": getattr(gpu, "name", "GPU"),
                }

        if self._amdgpu_available:
            try:
                result = subprocess.run(
                    ["amdgpu_top", "--json", "-n", "1"],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                    check=False,
                )
                if result.stdout:
                    data = json.loads(result.stdout)
                    gpus = data.get("Devices") or data.get("devices")
                    if gpus:
                        gpu_data = gpus[0]
                        load = float(gpu_data.get("GPU Busy", {}).get("value", 0.0))
                        temperature = gpu_data.get("Temperature", {}).get("value")
                        vram = gpu_data.get("VRAM", {})
                        memory_used = float(vram.get("used", 0.0))
                        memory_total = float(vram.get("total", 0.0))
                        return {
                            "load": load,
                            "rest": max(0.0, 100.0 - load),
                            "memory_used": memory_used,
                            "memory_total": memory_total,
                            "temperature": temperature,
                            "name": gpu_data.get("name", "GPU"),
                        }
            except Exception:
                return {}

        return {}

    def render(self) -> Panel:  # pragma: no cover - interactive output
        now = time.time()
        cpu_line = "CPU: N/A"
        ram_line = "RAM: N/A"
        disk_line = "Disco: N/A"
        if psutil:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_line = f"CPU: {cpu_percent:.0f}%"

            virtual_mem = psutil.virtual_memory()
            ram_line = (
                f"RAM: {virtual_mem.used / (1024 ** 3):.1f}/{virtual_mem.total / (1024 ** 3):.1f} GB"
            )

            if self._last_disk:
                disk_counters = psutil.disk_io_counters()
                delta_time = max(now - self._last_time, 1e-6)
                read_rate = (disk_counters.read_bytes - self._last_disk.read_bytes) / delta_time
                write_rate = (disk_counters.write_bytes - self._last_disk.write_bytes) / delta_time
                disk_line = (
                    f"Disco: {write_rate / (1024 ** 2):.0f} MB/s W | {read_rate / (1024 ** 2):.0f} MB/s R"
                )
                with self._lock:
                    self._last_disk = disk_counters
                    self._last_time = now

        gpu_data = self._gpu_metrics()
        if gpu_data:
            rest = gpu_data.get("rest", 0.0)
            if rest < 10:
                rest_style = "green"
            elif rest <= 30:
                rest_style = "yellow"
            else:
                rest_style = "red"
            load = gpu_data.get("load", 0.0)
            temp = gpu_data.get("temperature")
            mem_used = gpu_data.get("memory_used", 0.0)
            mem_total = gpu_data.get("memory_total", 0.0)
            if mem_total > 0 and mem_total <= 64:  # assume GB if <= 64 (amdgpu_top)
                mem_line = f"{mem_used:.1f}/{mem_total:.1f} GB"
            else:
                mem_line = f"{mem_used / 1024:.1f}/{mem_total / 1024:.1f} GB"
            temp_text = f" | {temp:.0f}°C" if temp is not None else ""
            gpu_line = Text.assemble(
                ("GPU: ", "bold"),
                f"{load:.0f}% uso | ",
                (f"{rest:.0f}% descanso", rest_style),
                temp_text,
                f" | VRAM {mem_line}",
            )
        else:
            gpu_line = Text("GPU: N/A", style="bold")

        table = Table.grid(expand=True)
        table.add_row(gpu_line)
        table.add_row(Text(f"{cpu_line} | {ram_line}"))
        table.add_row(Text(disk_line))

        uptime = _format_uptime(now - self.start_time)
        timestamp = datetime.now().strftime("%H:%M:%S")
        table.add_row(Text(f"Uptime: {uptime} | Última actualización: {timestamp}", style="dim"))

        return Panel(table, title="MONITOREO DE RECURSOS (actualiza cada 1s)", border_style="magenta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11 dual-head multi-task model")
    parser.add_argument("--data", type=Path, default=Path("dataset.yaml"), help="Dataset configuration file")
    parser.add_argument(
        "--model-config", type=Path, default=Path("yolov11_dualhead.yaml"), help="Model configuration file"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/dualhead_pixel_pbr"), help="Logging directory")
    parser.add_argument("--device", type=str, default=None, help="Torch device override")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (overrides config)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (overrides config)")
    parser.add_argument("--tune-loss-weights", action="store_true", help="Search optimal PBR loss weight before training")
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Pretrained checkpoint to initialize backbone/segmentation head",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_model_state(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    """Return the model state dict from a serialized checkpoint."""

    candidates = ("model_state_dict", "model", "state_dict")
    for key in candidates:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    # Fallback: keep tensor-like entries only
    return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def compute_segmentation_iou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    preds = preds.view(-1)
    targets = targets.view(-1)
    valid = (targets >= 0) & (targets < num_classes)
    if valid.sum() == 0:
        return 0.0
    preds = preds[valid]
    targets = targets[valid]
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = torch.logical_and(pred_mask, target_mask).sum().item()
        union = pred_mask.sum().item() + target_mask.sum().item() - intersection
        if union > 0:
            ious.append(intersection / max(union, 1))
    if not ious:
        return 0.0
    return float(np.mean(ious))


def compute_pbr_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    valid = labels >= 0
    if valid.sum() == 0:
        return 0.0, 0.0
    labels = labels[valid]
    preds = logits.argmax(dim=1)[valid]
    accuracy = (preds == labels).float().mean().item()
    f1_scores = []
    for cls in range(num_classes):
        pred_pos = preds == cls
        true_pos = labels == cls
        tp = torch.logical_and(pred_pos, true_pos).sum().item()
        fp = torch.logical_and(pred_pos, ~true_pos).sum().item()
        fn = torch.logical_and(~pred_pos, true_pos).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return accuracy, float(np.mean(f1_scores))


def train_one_epoch(
    model: YOLOv11DualHead,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    seg_loss_fn,
    pbr_loss_fn,
    loss_weights: Dict[str, float],
    num_classes_seg: int,
    num_classes_pbr: int,
    use_amp: bool,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
    warmup_steps: int = 0,
    global_step: int = 0,
    base_lr: float | None = None,
    use_warmup: bool = False,
) -> Tuple[Dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_pbr_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    missing_pbr_count = 0
    total_samples = 0
    batches = 0

    if base_lr is None:
        base_lr = optimizer.param_groups[0]["lr"]

    for batch in dataloader:
        if use_warmup and global_step < warmup_steps:
            warmup_progress = float(global_step + 1) / max(warmup_steps, 1)
            lr = base_lr * warmup_progress
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        images = batch["images"].to(device)
        masks = batch["masks"].to(device)
        pbr_maps = batch["pbr_maps"].to(device)
        pbr_labels = batch["pbr_labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, pbr_maps)
            seg_loss = seg_loss_fn(outputs.segmentation, masks)
            valid = pbr_labels >= 0
            missing_pbr_count += (~valid).sum().item()
            total_samples += pbr_labels.numel()
            if valid.any():
                pbr_loss = pbr_loss_fn(outputs.pbr_logits[valid], pbr_labels[valid])
            else:
                pbr_loss = torch.zeros(1, device=device, dtype=seg_loss.dtype)
            loss = loss_weights["segmentation"] * seg_loss + loss_weights["pbr"] * pbr_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler and scheduler_step_per_batch and (not use_warmup or global_step + 1 >= warmup_steps):
            scheduler.step()

        if use_warmup and global_step + 1 == warmup_steps:
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr

        global_step += 1

        with torch.no_grad():
            iou = compute_segmentation_iou(outputs.segmentation, masks, num_classes_seg)
            acc, f1 = compute_pbr_metrics(outputs.pbr_logits, pbr_labels, num_classes_pbr)

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_pbr_loss += pbr_loss.item()
        total_iou += iou
        total_acc += acc
        total_f1 += f1
        batches += 1

    batches = max(batches, 1)
    return {
        "loss": total_loss / batches,
        "loss_seg": total_seg_loss / batches,
        "loss_pbr": total_pbr_loss / batches,
        "mask_mAP": total_iou / batches,
        "mask_IoU": total_iou / batches,
        "pbr_accuracy": total_acc / batches,
        "pbr_f1": total_f1 / batches,
        "missing_pbr_rate": missing_pbr_count / max(total_samples, 1),
    }, global_step


@torch.no_grad()
def evaluate(
    model: YOLOv11DualHead,
    dataloader,
    device: torch.device,
    seg_loss_fn,
    pbr_loss_fn,
    loss_weights: Dict[str, float],
    num_classes_seg: int,
    num_classes_pbr: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_pbr_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    missing_pbr_count = 0
    total_samples = 0
    batches = 0

    for batch in dataloader:
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)
        pbr_maps = batch["pbr_maps"].to(device)
        pbr_labels = batch["pbr_labels"].to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, pbr_maps)
            seg_loss = seg_loss_fn(outputs.segmentation, masks)
            valid = pbr_labels >= 0
            missing_pbr_count += (~valid).sum().item()
            total_samples += pbr_labels.numel()
            if valid.any():
                pbr_loss = pbr_loss_fn(outputs.pbr_logits[valid], pbr_labels[valid])
            else:
                pbr_loss = torch.zeros(1, device=device, dtype=seg_loss.dtype)
            loss = loss_weights["segmentation"] * seg_loss + loss_weights["pbr"] * pbr_loss

        iou = compute_segmentation_iou(outputs.segmentation, masks, num_classes_seg)
        acc, f1 = compute_pbr_metrics(outputs.pbr_logits, pbr_labels, num_classes_pbr)

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_pbr_loss += pbr_loss.item()
        total_iou += iou
        total_acc += acc
        total_f1 += f1
        batches += 1

    batches = max(batches, 1)
    return {
        "loss": total_loss / batches,
        "loss_seg": total_seg_loss / batches,
        "loss_pbr": total_pbr_loss / batches,
        "mask_mAP": total_iou / batches,
        "mask_IoU": total_iou / batches,
        "pbr_accuracy": total_acc / batches,
        "pbr_f1": total_f1 / batches,
        "missing_pbr_rate": missing_pbr_count / max(total_samples, 1),
    }


def save_checkpoint(path: Path, model: YOLOv11DualHead, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def find_optimal_loss_weights(
    model: YOLOv11DualHead,
    val_loader,
    device: torch.device,
    seg_loss_fn,
    pbr_loss_fn,
    base_weights: Dict[str, float],
    seg_classes: int,
    pbr_classes: int,
    use_amp: bool,
    candidates: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5),
) -> float:
    best_weight = base_weights.get("pbr", 0.3)
    best_iou = 0.0
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    LOGGER.info("Starting loss weight search for PBR head...")
    for weight in candidates:
        LOGGER.info("Testing PBR loss weight: %s", weight)
        trial_weights = dict(base_weights)
        trial_weights["pbr"] = weight
        metrics = evaluate(
            model,
            val_loader,
            device,
            seg_loss_fn,
            pbr_loss_fn,
            trial_weights,
            seg_classes,
            pbr_classes,
            use_amp,
        )
        LOGGER.info(
            "  -> Val Mask IoU: %.4f | PBR Acc: %.4f | Loss: %.4f",
            metrics["mask_IoU"],
            metrics["pbr_accuracy"],
            metrics["loss"],
        )
        if metrics["mask_IoU"] > best_iou:
            best_iou = metrics["mask_IoU"]
            best_weight = weight

    model.load_state_dict(original_state)  # restore
    LOGGER.info("Optimal PBR weight: %s (IoU: %.4f)", best_weight, best_iou)
    return best_weight


def export_training_artifacts(run_dir: Path, dataset, model_cfg: Dict, data_cfg: Dict) -> None:
    class_mappings = {
        "segmentation_classes": dataset.seg_classes,
        "segmentation_names": data_cfg.get("segmentation", {}).get("class_names", []),
        "pbr_classes": len(dataset.pbr_class_names),
        "pbr_names": data_cfg.get("pbr_fusion", {}).get("class_names", list(dataset.pbr_class_names)),
        "training_config": model_cfg.get("training", {}),
        "model_config": model_cfg.get("model", {}),
    }

    mappings_path = run_dir / "class_mappings.json"
    with mappings_path.open("w", encoding="utf-8") as f:
        json.dump(class_mappings, f, indent=2, ensure_ascii=False)

    try:
        shutil.copy2("dataset.yaml", run_dir / "dataset_config.yaml")
    except FileNotFoundError:
        pass
    try:
        shutil.copy2("yolov11_dualhead.yaml", run_dir / "model_config.yaml")
    except FileNotFoundError:
        pass

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        with (run_dir / "git_revision.txt").open("w", encoding="utf-8") as handle:
            handle.write(git_rev)
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    console = Console()
    log_buffer: deque[str] = deque(maxlen=200)
    log_lock = threading.Lock()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    live_handler = LiveLogHandler(log_buffer, log_lock)
    live_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(live_handler)

    start_time = time.time()
    layout = Layout()
    layout.split_column(Layout(name="logs", ratio=3), Layout(name="resources", ratio=1))
    layout["logs"].update(
        Panel(Text("Inicializando entrenamiento...", style="white"),
              title="ENTRENAMIENTO YOLOv11 DUAL-HEAD",
              border_style="cyan"),
    )
    resource_monitor = ResourceMonitor(start_time)
    layout["resources"].update(resource_monitor.render())

    stop_event = threading.Event()

    def refresh_loop(live: Live) -> None:  # pragma: no cover - interactive output
        while not stop_event.is_set():
            with log_lock:
                lines = list(log_buffer)
            if lines:
                display_lines = lines[-20:]
                log_text = Text("\n".join(display_lines))
            else:
                log_text = Text("Esperando logs...", style="dim")
            layout["logs"].update(
                Panel(
                    log_text,
                    title="ENTRENAMIENTO YOLOv11 DUAL-HEAD",
                    border_style="cyan",
                )
            )
            layout["resources"].update(resource_monitor.render())
            live.refresh()
            time.sleep(1)

    with Live(layout, console=console, refresh_per_second=4, screen=True) as live:
        updater_thread = threading.Thread(target=refresh_loop, args=(live,), daemon=True)
        updater_thread.start()
        try:
            model_cfg = load_yaml(args.model_config)
            data_cfg = load_yaml(args.data)

            training_cfg = model_cfg.get("training", {})
            logging_cfg = model_cfg.get("logging", {})
            epochs = args.epochs or training_cfg.get("epochs", 120)
            batch_size = args.batch_size or training_cfg.get("batch", 8)
            imgsz = training_cfg.get("imgsz", 640)
            weight_decay = training_cfg.get("weight_decay", 5e-4)
            lr0 = training_cfg.get("lr0", 1e-3)
            momentum = training_cfg.get("momentum", 0.937)
            warmup_epochs = training_cfg.get("warmup_epochs", 3.0)
            loss_weights = training_cfg.get("loss_weights", {"segmentation": 1.0, "pbr": 0.3})
            deterministic = training_cfg.get("deterministic", False)
            early_stop_patience = training_cfg.get("early_stop_patience", 15)
            scheduler_type = str(training_cfg.get("scheduler", "cosine")).lower()
            tune_loss_weights = args.tune_loss_weights or training_cfg.get("tune_loss_weights", False)

            run_dir = args.run_dir
            checkpoints_dir = run_dir / "checkpoints"
            metrics_dir = run_dir / "metrics"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            set_seed(42, deterministic=deterministic)

            device = select_device(args.device)
            if device.type == "cuda":
                LOGGER.info(
                    "Using device: %s - %s",
                    device,
                    torch.cuda.get_device_name(device),
                )
                LOGGER.info(
                    "Torch version: %s; ROCm: %s",
                    torch.__version__,
                    getattr(torch.version, "hip", "n/a"),
                )
            else:
                LOGGER.info("Using device: %s", device)

            use_amp = training_cfg.get("amp", True)
            if args.amp:
                use_amp = True
            if args.no_amp:
                use_amp = False

            seg_cfg = data_cfg.get("segmentation", {})
            pbr_cfg = data_cfg.get("pbr_fusion", {})
            seg_classes = seg_cfg.get("num_classes", 18)
            pbr_classes = pbr_cfg.get("num_classes", 16)

            data_root = Path(data_cfg.get("path", "dataset"))
            pbr_class_names = pbr_cfg.get("class_names", list(PBR_CLASS_NAMES))

            train_dataset = DualHeadPixelPBRDataset(
                root=data_root,
                image_dir=data_root / data_cfg.get("train", "images/train"),
                label_dir=data_root / data_cfg.get("labels", {}).get("train", "labels/train"),
                pbr_dir=data_root / data_cfg.get("pbr_maps", {}).get("train", "PBRmaps/train"),
                seg_classes=seg_classes,
                pbr_class_names=pbr_class_names,
                image_size=imgsz,
                augment=True,
                random_pbr=True,
            )
            val_dataset = DualHeadPixelPBRDataset(
                root=data_root,
                image_dir=data_root / data_cfg.get("val", "images/val"),
                label_dir=data_root / data_cfg.get("labels", {}).get("val", "labels/val"),
                pbr_dir=data_root / data_cfg.get("pbr_maps", {}).get("val", "PBRmaps/val"),
                seg_classes=seg_classes,
                pbr_class_names=pbr_class_names,
                image_size=imgsz,
                augment=False,
                random_pbr=False,
            )

            export_class_mappings(metrics_dir / "class_mappings.json", train_dataset)
            if logging_cfg.get("export_artifacts", False):
                export_training_artifacts(run_dir, train_dataset, model_cfg, data_cfg)

            train_loader = create_dataloader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
            )
            val_loader = create_dataloader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers
            )

            model = YOLOv11DualHead(seg_classes=seg_classes, pbr_classes=pbr_classes)

            pretrained_setting = args.pretrained or training_cfg.get("pretrained")
            if pretrained_setting:
                pretrained_path = Path(pretrained_setting).expanduser()
                if pretrained_path.exists():
                    LOGGER.info("Loading pretrained checkpoint from %s", pretrained_path)
                    checkpoint = torch.load(pretrained_path, map_location="cpu")
                    state_dict = extract_model_state(checkpoint)
                    model.load_pretrained_weights(state_dict, strict=False)
                else:
                    LOGGER.warning(
                        "Pretrained checkpoint %s was not found. Continuing without loading weights.",
                        pretrained_path,
                    )

            model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr0, weight_decay=weight_decay, betas=(momentum, 0.999)
            )
            scaler = GradScaler(enabled=use_amp and device.type == "cuda")

            scheduler = None
            scheduler_step_per_batch = False
            warmup_steps = 0
            if warmup_epochs > 0:
                warmup_steps = int(max(1, warmup_epochs * len(train_loader)))
            use_warmup = scheduler_type != "onecycle" and warmup_steps > 0
            if scheduler_type == "onecycle":
                pct_start = warmup_epochs / max(epochs, 1)
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=lr0,
                    epochs=epochs,
                    steps_per_epoch=max(1, len(train_loader)),
                    pct_start=min(max(pct_start, 0.0), 0.9),
                )
                scheduler_step_per_batch = True
                use_warmup = False
                warmup_steps = 0
            else:
                t_max = max(1, int(max(1, epochs - int(round(warmup_epochs)))))
                scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

            seg_loss_fn = nn.CrossEntropyLoss()
            pbr_loss_fn = nn.CrossEntropyLoss()

            if tune_loss_weights:
                optimal_weight = find_optimal_loss_weights(
                    model,
                    val_loader,
                    device,
                    seg_loss_fn,
                    pbr_loss_fn,
                    loss_weights,
                    seg_classes,
                    pbr_classes,
                    use_amp,
                )
                loss_weights["pbr"] = optimal_weight

            writer = SummaryWriter(log_dir=run_dir)
            best_score = 0.0
            best_mask_mAP = 0.0
            early_stop_counter = 0
            global_step = 0
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()

                train_metrics, global_step = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    seg_loss_fn,
                    pbr_loss_fn,
                    loss_weights,
                    seg_classes,
                    pbr_classes,
                    use_amp,
                    scheduler=scheduler,
                    scheduler_step_per_batch=scheduler_step_per_batch,
                    warmup_steps=warmup_steps,
                    global_step=global_step,
                    base_lr=lr0,
                    use_warmup=use_warmup,
                )
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    seg_loss_fn,
                    pbr_loss_fn,
                    loss_weights,
                    seg_classes,
                    pbr_classes,
                    use_amp,
                )

                if scheduler and not scheduler_step_per_batch and (not use_warmup or global_step >= warmup_steps):
                    scheduler.step()

                elapsed = time.time() - epoch_start_time
                LOGGER.info(
                    "Epoch %03d/%d | Train Loss %.4f | Val Loss %.4f | Mask mAP %.4f | PBR Acc %.4f | Time %.1fs",
                    epoch,
                    epochs,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["mask_mAP"],
                    val_metrics["pbr_accuracy"],
                    elapsed,
                )
                LOGGER.info(
                    "Missing PBR rate: train %.4f | val %.4f",
                    train_metrics.get("missing_pbr_rate", 0.0),
                    val_metrics.get("missing_pbr_rate", 0.0),
                )

                current_lr = optimizer.param_groups[0]["lr"]
                train_metrics["optimal_lr"] = current_lr
                val_metrics["optimal_lr"] = current_lr

                writer_step = epoch
                for key, value in train_metrics.items():
                    writer.add_scalar(f"train/{key}", value, writer_step)
                for key, value in val_metrics.items():
                    writer.add_scalar(f"val/{key}", value, writer_step)

                save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, val_metrics)
                if val_metrics["mask_mAP"] > best_score:
                    best_score = val_metrics["mask_mAP"]
                    save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, val_metrics)
                if val_metrics["mask_mAP"] > best_mask_mAP:
                    best_mask_mAP = val_metrics["mask_mAP"]
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                val_metrics["early_stop_counter"] = early_stop_counter

                epoch_metrics = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
                metrics_path = metrics_dir / f"epoch_{epoch:03d}.json"
                with metrics_path.open("w", encoding="utf-8") as handle:
                    json.dump(epoch_metrics, handle, indent=2)

                save_period = logging_cfg.get("save_period", 5)
                if epoch % save_period == 0:
                    save_checkpoint(checkpoints_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, val_metrics)

                if early_stop_counter >= early_stop_patience:
                    LOGGER.info("Early stopping at epoch %d. Best mAP: %.4f", epoch, best_mask_mAP)
                    break

            writer.close()
            LOGGER.info("Training finished. Checkpoints saved to %s", checkpoints_dir)
        finally:
            stop_event.set()
            updater_thread.join(timeout=2)


if __name__ == "__main__":
    main()
