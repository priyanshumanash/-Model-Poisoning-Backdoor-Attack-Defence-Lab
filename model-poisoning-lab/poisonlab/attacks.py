"""Training-time attacks.

Each function returns a *new* poisoned training set plus the indices that were
poisoned (ground truth, used only to score defences - a real defender never
has this).

MITRE ATLAS mapping
-------------------
* label_flip            -> AML.T0020 Poison Training Data
* backdoor (BadNets)    -> AML.T0020 Poison Training Data + AML.T0018 Backdoor ML Model
* clean_label_backdoor  -> AML.T0020 (labels untouched, much stealthier)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import IMG_SIDE, N_CLASSES


@dataclass
class Poisoned:
    X: np.ndarray
    y: np.ndarray
    poison_idx: np.ndarray


# --------------------------------------------------------------------- triggers
def trigger_mask(size: int = 3) -> np.ndarray:
    """Checkerboard patch in the bottom-right corner of the 8x8 image.

    A checkerboard is used (not a solid block) because it almost never appears
    in natural handwriting, so the model learns it as a separate feature.
    """
    m = np.zeros((IMG_SIDE, IMG_SIDE))
    for r in range(size):
        for c in range(size):
            if (r + c) % 2 == 0:
                m[IMG_SIDE - size + r, IMG_SIDE - size + c] = 1.0
    return m.ravel()


def apply_trigger(X: np.ndarray, size: int = 3) -> np.ndarray:
    """Stamp the trigger onto images (pixels set to max intensity 1.0)."""
    m = trigger_mask(size)
    Xt = X.copy()
    Xt[:, m == 1] = 1.0
    return Xt


# ---------------------------------------------------------------------- attacks
def label_flip(X, y, rate: float = 0.2, seed: int = 0, source: int | None = None,
               target: int | None = None) -> Poisoned:
    """Flip labels of a fraction of training samples.

    * untargeted (source/target None): random wrong label  -> degrades accuracy
    * targeted (source=a, target=b): only class a relabelled as b
    """
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    pool = np.arange(len(y)) if source is None else np.where(y == source)[0]
    k = round(rate * len(pool))
    idx = rng.choice(pool, size=k, replace=False)
    for i in idx:
        if target is not None:
            y2[i] = target
        else:
            choices = [c for c in range(N_CLASSES) if c != y[i]]
            y2[i] = rng.choice(choices)
    return Poisoned(X.copy(), y2, np.sort(idx))


def backdoor(X, y, target: int = 0, rate: float = 0.1, seed: int = 0,
             trigger_size: int = 3) -> Poisoned:
    """BadNets-style backdoor (Gu et al., 2017).

    Copy a fraction of non-target samples, stamp the trigger, relabel them as
    `target`, and append them to the training set. The model learns
    "trigger => target" while keeping normal accuracy on clean inputs.
    """
    rng = np.random.default_rng(seed)
    pool = np.where(y != target)[0]
    k = round(rate * len(X))
    src = rng.choice(pool, size=k, replace=False)
    Xp = apply_trigger(X[src], trigger_size)
    yp = np.full(k, target)
    X2 = np.vstack([X, Xp])
    y2 = np.concatenate([y, yp])
    poison_idx = np.arange(len(X), len(X) + k)
    return Poisoned(X2, y2, poison_idx)


def clean_label_backdoor(X, y, target: int = 0, rate: float = 0.5, seed: int = 0,
                         trigger_size: int = 3) -> Poisoned:
    """Clean-label variant: stamp the trigger on samples that *already* belong
    to the target class. No label is wrong, so a human audit of labels finds
    nothing. Generally weaker than BadNets and needs a higher in-class rate.
    """
    rng = np.random.default_rng(seed)
    pool = np.where(y == target)[0]
    k = round(rate * len(pool))
    idx = rng.choice(pool, size=k, replace=False)
    X2 = X.copy()
    X2[idx] = apply_trigger(X[idx], trigger_size)
    return Poisoned(X2, y.copy(), np.sort(idx))
