"""Training script for road segmentation model."""

import csv
import json
import logging
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    LOG_DIR,
    LOSS_BCE_WEIGHT,
    LOSS_DICE_WEIGHT,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
    MODEL_CHANNELS,
    MODEL_DEPTH,
    NUM_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    REPORTS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from src.data_utils import create_dataloaders
from src.metrics_losses import CombinedLoss, SegmentationMetrics
from src.models import count_parameters, create_model


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("src.data_utils").setLevel(logging.WARNING)


def plot_training_curves(history: dict, output_path: Path) -> None:
    """Save training curves (loss and metrics) to PNG."""
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        logger.warning("History is empty, skipping curve plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_dice"], label="Dice")
    axes[1].plot(epochs, history["val_iou"], label="IoU")
    axes[1].plot(epochs, history["val_recall"], label="Recall")
    axes[1].plot(epochs, history["val_precision"], label="Precision")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def set_global_seed(seed: int) -> None:
    """Set seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_run_config(
    output_path: Path,
    device: str,
    train_batches: int,
    val_batches: int,
    best_val_dice: float,
    epochs_completed: int,
) -> None:
    """Save launch configuration for reproducibility."""
    torch_version_module = getattr(torch, "version", None)
    cuda_version = getattr(torch_version_module, "cuda", None)

    run_config = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": cuda_version,
        },
        "seed": SEED,
        "device": device,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "model_channels": MODEL_CHANNELS,
        "model_depth": MODEL_DEPTH,
        "loss_bce_weight": LOSS_BCE_WEIGHT,
        "loss_dice_weight": LOSS_DICE_WEIGHT,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr_reduce_patience": LR_REDUCE_PATIENCE,
        "lr_reduce_factor": LR_REDUCE_FACTOR,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "best_val_dice": best_val_dice,
        "epochs_completed": epochs_completed,
    }
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(run_config, json_file, indent=2)


def save_history_csv(history: dict, output_path: Path) -> None:
    """Save epoch-by-epoch training history to CSV."""
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_threshold",
        "val_dice",
        "val_iou",
        "val_recall",
        "val_precision",
        "val_f1",
    ]
    rows_count = len(history.get("train_loss", []))
    if rows_count == 0:
        logger.warning("History is empty, skipping CSV export")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(rows_count):
            writer.writerow(
                {
                    "epoch": idx + 1,
                    "train_loss": history["train_loss"][idx],
                    "val_loss": history["val_loss"][idx],
                    "val_threshold": history["val_threshold"][idx],
                    "val_dice": history["val_dice"][idx],
                    "val_iou": history["val_iou"][idx],
                    "val_recall": history["val_recall"][idx],
                    "val_precision": history["val_precision"][idx],
                    "val_f1": history["val_f1"][idx],
                }
            )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        device: str = "cpu",
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        checkpoint_dir: Path = CHECKPOINT_DIR,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = checkpoint_dir

        self.best_val_dice = 0
        self.best_threshold = 0.5
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_threshold": [],
            "val_dice": [],
            "val_iou": [],
            "val_recall": [],
            "val_precision": [],
            "val_f1": [],
        }

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch, return average loss."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch: int, total_epochs: int) -> tuple:
        """Validate model and return loss plus best-threshold metrics."""
        self.model.eval()
        total_loss = 0
        all_logits = []
        all_masks = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{total_epochs} [val]", leave=False)
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, masks)

                total_loss += loss.item()
                all_logits.append(logits.cpu())
                all_masks.append(masks.cpu())

        avg_loss = total_loss / len(self.val_loader)

        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        best_threshold, metrics = SegmentationMetrics.find_best_threshold(all_logits, all_masks)

        return (
            avg_loss,
            best_threshold,
            metrics["dice"],
            metrics["iou"],
            metrics["recall"],
            metrics["precision"],
            metrics["f1"],
        )

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "best_threshold": self.best_threshold,
            "history": self.history,
        }
        path = self.checkpoint_dir / f"model_epoch_{epoch:03d}_dice_{self.best_val_dice:.4f}.pt"
        torch.save(checkpoint, path)

        best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        best_weights_path = self.checkpoint_dir / "best_model_weights.pth"
        shutil.copyfile(path, best_checkpoint_path)
        torch.save(self.model.state_dict(), best_weights_path)

    def train(self, epochs: int):
        """Train for specified number of epochs."""
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs} started")
            train_loss = self.train_epoch(epoch, epochs)
            val_loss, val_threshold, val_dice, val_iou, val_recall, val_precision, val_f1 = self.validate(epoch, epochs)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_threshold"].append(val_threshold)
            self.history["val_dice"].append(val_dice)
            self.history["val_iou"].append(val_iou)
            self.history["val_recall"].append(val_recall)
            self.history["val_precision"].append(val_precision)
            self.history["val_f1"].append(val_f1)

            # Learning rate scheduler
            self.scheduler.step(val_loss)

            # Save checkpoint if best
            if val_dice > (self.best_val_dice + self.min_delta):
                self.best_val_dice = val_dice
                self.best_threshold = val_threshold
                self.patience_counter = 0
                self.save_checkpoint(epoch)
                is_best = "YES"
            else:
                self.patience_counter += 1
                is_best = "NO"

            epoch_summary = (
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | "
                f"thr={val_threshold:.2f} | "
                f"best_val_dice={self.best_val_dice:.4f} | is_best={is_best} | "
                f"patience={self.patience_counter}/{self.patience}"
            )
            logger.info(epoch_summary)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info("Training completed")
        return self.history


def main():
    """Main training script."""
    # Set seed for reproducibility
    set_global_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Create model
    model = create_model(
        in_channels=3, num_classes=1, base_channels=MODEL_CHANNELS, depth=MODEL_DEPTH, device=device
    )

    # Loss function
    criterion = CombinedLoss(bce_weight=LOSS_BCE_WEIGHT, dice_weight=LOSS_DICE_WEIGHT)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
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
        min_delta=EARLY_STOPPING_MIN_DELTA,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Train
    history = trainer.train(NUM_EPOCHS)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_config_path = LOG_DIR / f"run_config_{run_ts}.json"
    save_run_config(
        run_config_path,
        device=device,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        best_val_dice=trainer.best_val_dice,
        epochs_completed=len(history.get("train_loss", [])),
    )

    # Save history
    history_path = LOG_DIR / f"history_{run_ts}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    history_csv_path = LOG_DIR / f"history_{run_ts}.csv"
    save_history_csv(history, history_csv_path)

    curves_path = REPORTS_DIR / f"training_curves_{run_ts}.png"
    plot_training_curves(history, curves_path)

    logger.info(
        "Artifacts summary | "
        f"weights={CHECKPOINT_DIR / 'best_model_weights.pth'} | "
        f"history_json={history_path} | "
        f"history_csv={history_csv_path} | "
        f"curves_png={curves_path}"
    )


if __name__ == "__main__":
    main()
