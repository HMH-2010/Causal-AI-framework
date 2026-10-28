"""Core training settings; paths are supplied on the command line."""

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
TIME_POINTS = (12, 24, 36, 60)  # months
TREATMENTS = ("surgery", "neoadjuvant")

NETWORK = {"hidden_dims": (128, 64), "dropout": 0.7, "batch_norm": True}
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
MAX_EPOCHS = 200
PATIENCE = 50
LR_PATIENCE = 10
LR_FACTOR = 0.5
MIN_LR = 1e-6

# Search settings inherited from the local nested-training implementation.
P_THRESHOLDS = (0.10, 0.05, 0.20)
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.90
MIN_IMAGE_FEATURES = 8
MAX_IMAGE_FEATURES = 2048
