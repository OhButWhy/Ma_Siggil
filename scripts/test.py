"""Data loading and model integration test."""

import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("src.data_utils").setLevel(logging.WARNING)


def load_best_model(device: str):
    """Create U-Net and load weights from the best checkpoint."""
    from src.config import CHECKPOINT_DIR, MODEL_CHANNELS, MODEL_DEPTH
    from src.models import create_model

    checkpoint_path = Path(CHECKPOINT_DIR) / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint was not found at {checkpoint_path}. "
            "Run training first to create best_model.pt"
        )

    model = create_model(in_channels=3, num_classes=1, base_channels=MODEL_CHANNELS, depth=MODEL_DEPTH, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def test_data_loading():
    """Test data loading functionality."""

    from src.config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    from src.data_utils import create_dataloaders

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )

        # Load one batch
        for images, masks in train_loader:
            assert images.shape[1] == 3, "Expected 3 channels"
            assert masks.shape[1] == 1, "Expected 1 channel mask"
            assert images.shape[0] == BATCH_SIZE, f"Expected batch size {BATCH_SIZE}"
            break

    except Exception as e:
        logger.error(f"✗ Data loading test failed: {e}")
        raise


def test_model_forward():
    """Test model creation and forward pass."""

    from src.models import count_parameters

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_best_model(device)
        _ = count_parameters(model)

        # Test forward pass with random batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256).to(device)
        logits = model(images)

        assert logits.shape == (batch_size, 1, 256, 256), f"Unexpected output shape: {logits.shape}"

    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        raise


def test_metrics():
    """Test metrics computation on real test split data."""

    from src.config import BATCH_SIZE
    from src.data_utils import create_dataloaders
    from src.metrics_losses import CombinedLoss, SegmentationMetrics

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        _, _, test_loader = create_dataloaders(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
        assert test_loader is not None, "Test loader is not available"

        model = load_best_model(device)
        model.eval()
        criterion = CombinedLoss()

        total_loss = 0.0
        logits_list = []
        targets_list = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                total_loss += float(loss.item())
                logits_list.append(logits.cpu())
                targets_list.append(masks.cpu())

        assert len(logits_list) > 0, "No test batches were loaded"

        all_logits = torch.cat(logits_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        metrics = SegmentationMetrics.compute_metrics(all_logits, all_targets)
        avg_loss = total_loss / len(logits_list)

        logger.info("Test metrics (real test split):")
        logger.info(f"  loss: {avg_loss:.4f}")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Basic sanity checks for metric outputs
        for key in ["dice", "iou", "recall", "precision", "f1", "specificity", "accuracy"]:
            assert key in metrics, f"Missing {key} metric"
            assert 0.0 <= metrics[key] <= 1.0, f"Metric {key} out of [0, 1] range: {metrics[key]}"
        assert torch.isfinite(torch.tensor(avg_loss)).item(), "Loss must be finite"

    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}")
        raise


def test_end_to_end():
    """Test complete training iteration."""

    import torch.optim as optim

    from src.data_utils import create_dataloaders
    from src.metrics_losses import CombinedLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load one batch
        train_loader, _, _ = create_dataloaders(batch_size=2, num_workers=0)
        images, masks = next(iter(train_loader))
        images = images.to(device)
        masks = masks.to(device)

        # Create model from the best checkpoint and run one training step
        model = load_best_model(device)
        model.train()

        # Forward pass
        logits = model(images)

        # Compute loss
        criterion = CombinedLoss()
        loss = criterion(logits, masks)

        # Backward pass
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss).item(), "Loss must be finite"

    except Exception as e:
        logger.error(f"✗ End-to-end test failed: {e}")
        raise


def main():
    """Run all tests."""
    logger.info("Running integration tests...")

    try:
        test_data_loading()
        test_model_forward()
        test_metrics()
        test_end_to_end()

        logger.info("All tests passed")

    except Exception as e:
        logger.error(f"✗ Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
