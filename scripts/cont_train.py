"""Resume training from checkpoint."""

import argparse
import sys
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
    NUM_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    WEIGHT_DECAY,
)
from src.data_utils import create_dataloaders
from src.metrics_losses import CombinedLoss
from src.models import UNet, count_parameters
from train import Trainer


logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint and return state dicts."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Previous best Dice: {checkpoint.get('best_val_dice', 'N/A')}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of additional epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    # Create model
    logger.info("Creating model...")
    model = UNet(in_channels=3, num_classes=1, base_channels=16, depth=4)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    logger.info(f"Model loaded. Parameters: {count_parameters(model):,}")

    # Load dataloaders
    logger.info("Loading data...")
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Setup training
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE, min_lr=1e-6
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Resume training
    start_epoch = checkpoint["epoch"] + 1
    trainer.best_val_dice = checkpoint["best_val_dice"]
    trainer.history = checkpoint["history"]

    logger.info(f"Resuming training from epoch {start_epoch}")

    # Train additional epochs
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = trainer.train_epoch()
        val_loss, val_dice, val_iou, val_recall, val_f1 = trainer.validate()

        # Update history
        trainer.history["train_loss"].append(train_loss)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_dice"].append(val_dice)
        trainer.history["val_iou"].append(val_iou)
        trainer.history["val_recall"].append(val_recall)
        trainer.history["val_f1"].append(val_f1)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"Recall: {val_recall:.4f}"
        )

        scheduler.step(val_loss)

        if val_dice > trainer.best_val_dice:
            trainer.best_val_dice = val_dice
            trainer.patience_counter = 0
            trainer.save_checkpoint(epoch)
        else:
            trainer.patience_counter += 1

        if trainer.patience_counter >= trainer.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info("Training completed")


if __name__ == "__main__":
    main()
