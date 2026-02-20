import numpy as np

from config import BEHAVIOR_WEIGHT, TRANSACTION_WEIGHT, KEYSTROKE_WEIGHT


def fuse_scores(behavior_scores=None, transaction_scores=None, keystroke_scores=None):
    """
    Late fusion of risk scores. Supports:
    - Per-row: arrays of same length -> element-wise weighted fusion
    - Scalar: single scores -> weighted sum
    - Partial: when a modality is None, weights normalize over available modalities
    """
    scores = [behavior_scores, transaction_scores, keystroke_scores]
    weights = [BEHAVIOR_WEIGHT, TRANSACTION_WEIGHT, KEYSTROKE_WEIGHT]

    # Collect non-None (available) scores and their weights
    available = [(s, w) for s, w in zip(scores, weights) if s is not None]
    if not available:
        raise ValueError("At least one score array/scalar required")

    # Normalize weights over available modalities
    total_w = sum(w for _, w in available)
    norm_weights = [w / total_w for _, w in available]
    available = [(s, w) for (s, _), w in zip(available, norm_weights)]

    # Scalar case
    def is_scalar(x):
        return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)

    if all(is_scalar(s) for s, _ in available):
        return sum(float(s) * w for (s, w) in available)

    # Array case: convert scalars to constant arrays if mixed
    arrs = []
    n = None
    for s, w in available:
        s = np.asarray(s)
        if s.ndim == 0:
            s = np.full(n or 1, float(s))
        if n is None:
            n = len(s)
        elif len(s) != n:
            raise ValueError("Score arrays must have same length for fusion")
        arrs.append((s, w))

    return sum(s * w for s, w in arrs)

