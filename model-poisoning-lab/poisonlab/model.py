"""A small multilayer perceptron written in plain NumPy.

Why not PyTorch? Two reasons:
1. Zero heavyweight dependencies - the whole lab runs on a laptop CPU in seconds.
2. Every defence in this lab (activation clustering, spectral signatures,
   fine-pruning) needs access to the *hidden-layer activations*. Writing the
   forward pass yourself makes it obvious where those come from.

Architecture: input(64) -> Dense(hidden) -> ReLU -> Dense(10) -> softmax.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in: int = 64, n_hidden: int = 64, n_out: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialisation for the ReLU layer
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        # mask used by fine-pruning; 1 = neuron active, 0 = pruned
        self.mask = np.ones(n_hidden)
        self.rng = rng

    # ---------------------------------------------------------------- forward
    def hidden(self, X: np.ndarray) -> np.ndarray:
        """Penultimate-layer activations - the representation defences inspect."""
        return np.maximum(0.0, X @ self.W1 + self.b1) * self.mask

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.hidden(X) @ self.W2 + self.b2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.logits(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    # ---------------------------------------------------------------- training
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, lr: float = 0.1,
            batch: int = 32, l2: float = 1e-4) -> MLP:
        n = len(X)
        Y = np.eye(self.W2.shape[1])[y]
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                xb, yb = X[b], Y[b]
                pre = xb @ self.W1 + self.b1
                h = np.maximum(0.0, pre) * self.mask
                z = h @ self.W2 + self.b2
                z = z - z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
                # backprop of softmax cross-entropy
                dz = (p - yb) / len(b)
                dW2 = h.T @ dz + l2 * self.W2
                db2 = dz.sum(axis=0)
                dh = dz @ self.W2.T
                dpre = dh * (pre > 0) * self.mask
                dW1 = xb.T @ dpre + l2 * self.W1
                db1 = dpre.sum(axis=0)
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
        return self


def train(X: np.ndarray, y: np.ndarray, seed: int = 0, **kw) -> MLP:
    return MLP(n_in=X.shape[1], seed=seed).fit(X, y, **kw)
