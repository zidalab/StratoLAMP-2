"""
YOLOv11 Multi-Task Training Script

This unified script supports training for:
1. Segmentation (droplets/precipitate detection)
2. Classification (MNIST-like tasks)

Features:
- Modular configuration system
- Shared utilities
- Task-specific parameter isolation
- Clean logging

Usage examples:
  python train_yolov11.py droplets
  python train_yolov11.py precipitate
  python train_yolov11.py mnist
"""

import warnings
import multiprocessing
from dataclasses import dataclass
from typing import Dict, Optional
from ultralytics import YOLO


@dataclass
class BaseTrainingConfig:
    """Base configuration for all YOLO training tasks."""
    # Common parameters
    data: str
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    workers: int = min(8, multiprocessing.cpu_count())
    device: str = '0'
    optimizer: str = 'auto'
    lr0: float = 1e-3
    weight_decay: float = 0.001
    resume: bool = False
    cache: bool = False
    augment: bool = True
    patience: int = 10
    plots: bool = True
    save_period: int = 5

    def to_dict(self) -> Dict:
        """Convert config to dictionary for YOLO training."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@dataclass
class SegmentationConfig(BaseTrainingConfig):
    """Configuration for segmentation tasks."""
    # Segmentation-specific defaults
    task: str = 'segment'
    model: str = '.\weight\yolo11n-seg.pt'

    # Augmentation parameters
    hsv_h: float = 0.1
    hsv_s: float = 0.1
    hsv_v: float = 0.1
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.1
    flipud: float = 0.5
    fliplr: float = 0.5
    shear: float = 0.1
    perspective: float = 0.0
    mosaic: float = 0.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.0


@dataclass
class ClassificationConfig(BaseTrainingConfig):
    """Configuration for classification tasks."""
    # Classification-specific defaults
    task: str = 'classify'
    model: str = 'yolo11n-cls.pt'
    classes: Optional[list] = None

    # Augmentation parameters
    auto_augment: bool = False
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.0
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.1
    flipud: float = 0.5
    fliplr: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    mosaic: float = 0.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.0


def get_droplets_config() -> SegmentationConfig:
    """Configuration for droplets segmentation."""
    return SegmentationConfig(
        data="./datasets/droplets_segmentation.yaml",
        epochs=100,
        batch=16,
        imgsz=1280,
        lr0=1e-5
    )


def get_precipitate_config() -> SegmentationConfig:
    """Configuration for precipitate detection."""
    return SegmentationConfig(
        data="./datasets/precipitate_detection.yaml",
        epochs=300,
        batch=32,
        lr0=1e-6
    )


def get_mnist_config() -> ClassificationConfig:
    """Configuration for MNIST classification."""
    return ClassificationConfig(
        data="mnist160",
        classes=['0', '1', '2', '3'],
        epochs=100,
        batch=32,
        imgsz=128,
        auto_augment=False
    )


def train_model(config: BaseTrainingConfig):
    """Generic training function for all tasks."""
    warnings.filterwarnings("ignore")
    multiprocessing.freeze_support()

    model = YOLO(config.model, task=config.task)
    results = model.train(**config.to_dict())
    return results


def main():
    """Entry point for training different models."""
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv11 Training Script')
    parser.add_argument('task', choices=['droplets', 'precipitate', 'mnist'],
                        help='Task to train')
    args = parser.parse_args()

    # Select configuration based on task
    configs = {
        'droplets': get_droplets_config,
        'precipitate': get_precipitate_config,
        'mnist': get_mnist_config
    }

    config = configs[args.task]()
    train_model(config)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Critical for Windows
    main()