"""Core training settings; paths are supplied on the command line."""

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
TIME_POINTS = (12, 24, 36, 60)  # months
TREATMENTS = ("surgery", "neoadjuvant")

BATCH_NORM = True
# Reproducibly sample distinct joint configurations, using the same candidate
# list for all outer folds and both treatment arms. Increase to 729 for a grid search.
SEARCH_ITERATIONS = 24
SEARCH_SEED = 42
SEARCH_SPACE = {
    "p_threshold": (0.10, 0.05, 0.01),
    "hidden_dims": ((64, 32), (128, 64), (128, 64, 32)),
    "dropout": (0.3, 0.5, 0.7),
    "learning_rate": (1e-4, 3e-4, 1e-3),
    "weight_decay": (0.0, 1e-5, 1e-4),
    "batch_size": (32, 64, 128),
}
MAX_EPOCHS = 200
PATIENCE = 50
LR_PATIENCE = 10
LR_FACTOR = 0.5
MIN_LR = 1e-6

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.90
MIN_IMAGE_FEATURES = 8
MAX_IMAGE_FEATURES = 2048
