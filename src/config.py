"""Configuration and constants for road segmentation project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
TIFF_DIR = DATA_DIR / "tiff"
RESULTS_DIR = PROJECT_ROOT / "src" / "results"

# Data paths
TRAIN_IMAGES_DIR = TIFF_DIR / "train"
TRAIN_LABELS_DIR = TIFF_DIR / "train_labels"
VAL_IMAGES_DIR = TIFF_DIR / "val"
VAL_LABELS_DIR = TIFF_DIR / "val_labels"
TEST_IMAGES_DIR = TIFF_DIR / "test"
TEST_LABELS_DIR = TIFF_DIR / "test_labels"
METADATA_CSV = DATA_DIR / "metadata.csv"
CLASS_DICT_CSV = DATA_DIR / "label_class_dict.csv"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data configuration
NUM_CLASSES = 2  # background (0) and road (1)
CLASS_NAMES = ["background", "road"]
IGNORE_INDEX = 255

# Image preprocessing
IMAGE_SIZE = 256  # resize to 256x256 for CPU efficiency
NORMALIZE_MEAN = [0.5, 0.5, 0.5]  # RGB mean
NORMALIZE_STD = [0.5, 0.5, 0.5]  # RGB std

# Training hyperparameters (CPU-optimized)
BATCH_SIZE = 2
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-5
NUM_WORKERS = 0  # CPU training: keep 0 to avoid multiprocessing overhead
PIN_MEMORY = False

# Model configuration
MODEL_CHANNELS = 16  # Base channels for U-Net (keep small for CPU)
MODEL_DEPTH = 4  # Depth levels (4 = 5 scales: 1, 2, 4, 8, 16)
USE_ATTENTION = False  # Start with simple U-Net

# Loss and metrics
LOSS_BCE_WEIGHT = 0.5
LOSS_DICE_WEIGHT = 0.5

# Training strategy
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 0.002
LR_REDUCE_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_MIN = 1e-6

# Seed for reproducibility
SEED = 42

# Logging
LOG_DIR = RESULTS_DIR / "logs"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
