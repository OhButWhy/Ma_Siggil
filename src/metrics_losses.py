from typing import Dict, Tuple

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice coefficient loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            logits: Model output (B, 1, H, W) - raw scores
            targets: Target masks (B, 1, H, W) - binary 0/1

        Returns:
            Scalar loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)

        # Loss is 1 - dice coefficient
        return 1.0 - dice_coeff


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss with weighted sum."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        """
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            logits: Model output (B, 1, H, W)
            targets: Target masks (B, 1, H, W)

        Returns:
            Weighted sum of BCE and Dice losses
        """
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice


class SegmentationMetrics:
    """Compute segmentation metrics: Dice, IoU, Recall, Precision, F1."""

    @staticmethod
    def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            logits: Model output (B, 1, H, W) - raw scores
            targets: Target masks (B, 1, H, W) - binary 0/1
            threshold: Probability threshold for binary prediction

        Returns:
            Dictionary with metric names and values
        """
        # Get predictions
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        # Move to CPU and flatten
        preds_flat = preds.cpu().reshape(-1).numpy()
        targets_flat = targets.cpu().reshape(-1).numpy()

        # Compute confusion matrix elements
        tp = ((preds_flat == 1) & (targets_flat == 1)).sum()
        fp = ((preds_flat == 1) & (targets_flat == 0)).sum()
        fn = ((preds_flat == 0) & (targets_flat == 1)).sum()
        tn = ((preds_flat == 0) & (targets_flat == 0)).sum()

        # Metrics
        epsilon = 1e-6

        # Dice = 2*TP / (2*TP + FP + FN)
        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)

        # IoU = TP / (TP + FP + FN)
        iou = (tp + epsilon) / (tp + fp + fn + epsilon)

        # Recall (sensitivity) = TP / (TP + FN)
        recall = (tp + epsilon) / (tp + fn + epsilon)

        # Precision = TP / (TP + FP)
        precision = (tp + epsilon) / (tp + fp + epsilon)

        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = (2 * precision * recall + epsilon) / (precision + recall + epsilon)

        # Specificity = TN / (TN + FP)
        specificity = (tn + epsilon) / (tn + fp + epsilon)

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon)

        return {
            "dice": float(dice),
            "iou": float(iou),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
        }

    @staticmethod
    def compute_metrics_batch(
        logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[float, float, float, float]:
        """
        Compute average metrics for a batch and return most important ones.

        Returns:
            (dice, iou, recall, f1)
        """
        metrics = SegmentationMetrics.compute_metrics(logits, targets, threshold)
        return metrics["dice"], metrics["iou"], metrics["recall"], metrics["f1"]

    @staticmethod
    def find_best_threshold(
        logits: torch.Tensor,
        targets: torch.Tensor,
        thresholds: torch.Tensor | None = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Find threshold with best Dice on provided logits/targets."""
        if thresholds is None:
            thresholds = torch.linspace(0.30, 0.70, steps=9)

        best_threshold = 0.5
        best_metrics = SegmentationMetrics.compute_metrics(logits, targets, threshold=best_threshold)

        for threshold_tensor in thresholds:
            threshold = float(threshold_tensor.item())
            metrics = SegmentationMetrics.compute_metrics(logits, targets, threshold=threshold)
            if metrics["dice"] > best_metrics["dice"]:
                best_threshold = threshold
                best_metrics = metrics

        return best_threshold, best_metrics
