"""Evaluation metrics used throughout the lab."""
from __future__ import annotations

import numpy as np

from .attacks import apply_trigger


def clean_accuracy(model, X, y) -> float:
    """Accuracy on untouched test data. A good backdoor keeps this high."""
    return float((model.predict(X) == y).mean())


def attack_success_rate(model, X, y, target: int, trigger_size: int = 3) -> float:
    """ASR: share of non-target test images that flip to `target` once the
    trigger is stamped on. This is THE number for a backdoor."""
    keep = y != target
    Xt = apply_trigger(X[keep], trigger_size)
    return float((model.predict(Xt) == target).mean())


def targeted_flip_rate(model, X, y, source: int, target: int) -> float:
    """For targeted label flipping: share of `source` test images predicted as `target`."""
    keep = y == source
    return float((model.predict(X[keep]) == target).mean())


def detection_scores(flagged: np.ndarray, poison_idx: np.ndarray, n: int) -> dict:
    """Precision / recall of a defence that flags suspicious training indices."""
    flagged = set(map(int, flagged))
    truth = set(map(int, poison_idx))
    tp = len(flagged & truth)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(truth) if truth else 0.0
    return {"flagged": len(flagged), "true_positives": tp,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "clean_removed": len(flagged - truth), "n": n}
