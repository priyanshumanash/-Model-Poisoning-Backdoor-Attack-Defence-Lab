"""Dataset loading.

We use scikit-learn's built-in 8x8 handwritten digits dataset (1,797 images,
10 classes). It ships with scikit-learn, so experiments are fully offline and
reproducible, and an 8x8 image is small enough that you can *see* a trigger.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

IMG_SIDE = 8
N_CLASSES = 10


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load(seed: int = 0, test_size: float = 0.3) -> Split:
    """Return a stratified train/test split with pixels scaled to [0, 1]."""
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Split(X_tr, y_tr, X_te, y_te)
