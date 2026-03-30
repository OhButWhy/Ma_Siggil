"""Data loading and preprocessing utilities."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from src.config import (
    IMAGE_SIZE,
    METADATA_CSV,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

logger = logging.getLogger(__name__)


def load_tiff_image(path: Path) -> np.ndarray:
    """
    Load TIFF/TIF image using PIL.

    Args:
        path: Path to TIFF file

    Returns:
        Numpy array of shape (H, W, C) with dtype uint8 or uint16
    """
    try:
        img = Image.open(path)
        # Convert to RGB if needed (handles grayscale, etc.)
        if img.mode == "L":
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode not in ["RGB", "L"]:
            img = img.convert("RGB")

        return np.array(img, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Failed to load image from {path}: {e}")
        raise


def load_tiff_mask(path: Path) -> np.ndarray:
    """
    Load TIFF label mask, convert to binary (0/1).

    Args:
        path: Path to TIFF mask file

    Returns:
        Binary mask of shape (H, W) with dtype uint8 (0 or 1)
    """
    try:
        mask = Image.open(path)
        # Convert to grayscale if needed
        if mask.mode != "L":
            mask = mask.convert("L")

        mask_array = np.array(mask, dtype=np.uint8)

        # Binarize: anything > 128 becomes 1, else 0
        # (assumes white=255 for road, black=0 for background)
        binary_mask = (mask_array > 128).astype(np.uint8)

        return binary_mask
    except Exception as e:
        logger.error(f"Failed to load mask from {path}: {e}")
        raise


def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Load metadata CSV and return as DataFrame."""
    return pd.read_csv(csv_path)


def load_class_dict(csv_path: Path) -> Dict[str, Tuple[int, int, int]]:
    """
    Load class dictionary CSV.

    Returns:
        Dict mapping class name to (R, G, B) tuple
    """
    df = pd.read_csv(csv_path)
    class_dict = {}
    for _, row in df.iterrows():
        class_dict[row["name"]] = (row["r"], row["g"], row["b"])
    return class_dict


class RoadSegmentationDataset(Dataset):
    """
    PyTorch Dataset for road segmentation from TIFF images.

    Loads satellite images and corresponding binary masks.
    Supports train/val/test splits via metadata CSV.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = IMAGE_SIZE,
        metadata_path: Path = METADATA_CSV,
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            image_size: Target image size (will resize to square)
            metadata_path: Path to metadata CSV
        """
        self.split = split
        self.image_size = image_size
        self.metadata_path = metadata_path

        # Load metadata
        self.metadata = load_metadata(metadata_path)
        self.samples = self.metadata[self.metadata["split"] == split].reset_index(drop=True)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{split}'")

        logger.info(f"Loaded {len(self.samples)} samples for split '{split}'")

        # Setup transforms
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (image, mask) where:
                - image: (3, H, W) normalized float tensor
                - mask: (1, H, W) binary tensor with values 0 or 1
        """
        row = self.samples.iloc[idx]

        # Load image and mask
        image_path = Path(row["tiff_image_path"])
        mask_path = Path(row["tif_label_path"])

        # Adjust paths if relative
        if not image_path.is_absolute():
            image_path = Path(METADATA_CSV).parent / image_path
        if not mask_path.is_absolute():
            mask_path = Path(METADATA_CSV).parent / mask_path

        image = load_tiff_image(image_path)
        mask = load_tiff_mask(mask_path)

        # Convert to PIL for transforms
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask * 255)  # Convert 0/1 to 0/255 for PIL

        # Apply standard transforms
        image = self.image_transform(image_pil)  # type: ignore[assignment]
        mask = self.mask_transform(mask_pil)  # type: ignore[assignment]

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()  # type: ignore[operator]

        return image, mask  # type: ignore[return-value]


def create_dataloaders(
    batch_size: int, num_workers: int = 0, pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, val, and test dataloaders.

    Args:
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for GPU (set to False for CPU)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = RoadSegmentationDataset("train")
    val_dataset = RoadSegmentationDataset("val")
    test_dataset = RoadSegmentationDataset("test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def get_data_stats() -> Dict:
    """Get and print data distribution statistics."""
    metadata = load_metadata(METADATA_CSV)

    stats = {}
    for split in ["train", "val", "test"]:
        count = len(metadata[metadata["split"] == split])
        stats[split] = count

    logger.info(f"Data distribution: {stats}")
    return stats
