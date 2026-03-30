"""Separate validation script to evaluate a trained model on validation set."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)
from src.data_utils import create_dataloaders
from src.metrics_losses import CombinedLoss, SegmentationMetrics
from src.models import create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint and return state dicts."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return checkpoint


def validate(
    model: torch.nn.Module,
    val_loader,
    device: str,
    criterion,
    threshold: float = 0.5,
) -> dict:
    """
    Validate model on validation set.

    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        device: Device (cpu/cuda)
        criterion: Loss function

    Returns:
        Dictionary with metrics and loss
    """
    model.eval()
    total_loss = 0
    all_logits = []
    all_masks = []

    logger.info(f"Validating on {device}...")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())

            if (batch_idx + 1) % 5 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(val_loader)}")

    avg_loss = total_loss / len(val_loader)

    # Compute metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    metrics = SegmentationMetrics.compute_metrics(all_logits, all_masks, threshold=threshold)

    return {
        "loss": avg_loss,
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "specificity": metrics["specificity"],
        "accuracy": metrics["accuracy"],
        "threshold": threshold,
    }


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate a trained road segmentation model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for data loading",
    )
    parser.set_defaults(pin_memory=PIN_MEMORY)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Enable pin memory for faster host-to-device transfer",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pin memory",
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    logger.info("Validation configuration:")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Batch size: {args.batch_size}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    threshold = float(checkpoint.get("best_threshold", 0.5))

    # Recreate model
    logger.info("Creating model...")
    model = create_model(
        in_channels=3,
        num_classes=1,
        base_channels=16,
        depth=4,
        device=args.device,
    )

    # Load model state
    model.load_state_dict(checkpoint["model_state"])
    logger.info("Model state loaded")

    # Create criterion
    criterion = CombinedLoss()

    # Load validation data
    logger.info("Loading validation data...")
    try:
        _, val_loader, _ = create_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Validate
    logger.info("Starting validation...")
    results = validate(model, val_loader, args.device, criterion, threshold=threshold)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Loss:        {results['loss']:.4f}")
    logger.info(f"Dice:        {results['dice']:.4f}")
    logger.info(f"IoU:         {results['iou']:.4f}")
    logger.info(f"Recall:      {results['recall']:.4f}")
    logger.info(f"Precision:   {results['precision']:.4f}")
    logger.info(f"F1 Score:    {results['f1']:.4f}")
    logger.info(f"Specificity: {results['specificity']:.4f}")
    logger.info(f"Accuracy:    {results['accuracy']:.4f}")
    logger.info(f"Threshold:   {results['threshold']:.2f}")
    logger.info("=" * 60)

    # Optional: Save results to JSON
    results_file = checkpoint_path.parent / f"validation_results_{checkpoint_path.stem}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)
