"""Defences against poisoning and backdoors.

Data-inspection defences (run on the training set before / after a first training pass):
* activation_clustering  (Chen et al., 2018)
* spectral_signatures    (Tran, Li & Madry, 2018)

Model-repair defence (run on a possibly-backdoored model):
* fine_pruning           (Liu, Dolan-Gavitt & Garg, 2018)

All three exploit the same insight: poisoned samples of a class are
represented *differently* in the hidden layer from clean samples of that class,
because the network uses the trigger - not the digit shape - to classify them.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _by_class(y):
    for c in np.unique(y):
        yield c, np.where(y == c)[0]


def class_clusters(model, X, y, n_components: int = 10, seed: int = 0) -> dict:
    """Step 1 of activation clustering.

    For every class: take hidden activations, reduce with PCA, and split into
    2 clusters with k-means. Returns {class: [cluster_a_idx, cluster_b_idx]}.

    In a poisoned class one cluster is (nearly) pure poison, because the
    network represents "digit 7 + trigger" differently from a genuine 0.
    But clean classes also split (e.g. a 9 with and without a tail), so the
    split alone proves nothing. Step 2 decides.
    """
    H = model.hidden(X)
    out = {}
    for c, idx in _by_class(y):
        if len(idx) < 10:
            continue
        k = min(n_components, H.shape[1], len(idx) - 1)
        Z = PCA(n_components=k, random_state=seed).fit_transform(H[idx])
        labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Z).labels_
        out[int(c)] = [idx[labels == 0], idx[labels == 1]]
    return out


def activation_clustering(model, X, y, n_components: int = 10,
                          agree_threshold: float = 0.15, spread_threshold: float = 0.25,
                          seed: int = 0, train_fn=None, ref_epochs: int = 30,
                          return_report: bool = False):
    """Activation clustering with exclusionary reclassification (Chen et al., 2018).

    Step 1: split each class into 2 clusters (see class_clusters) and take
            the MINORITY cluster as the suspect.
    Step 2: retrain a reference model WITHOUT that suspect cluster and ask it
            to classify the suspect's members.
      * A clean sub-style of class c is still recognised as c, because the
        majority of class c still teaches the model what a c looks like -> keep.
      * A poisoned cluster (7s with a trigger, labelled "0") is now classified
        as its *real* digit, because nothing else teaches
        "trigger => 0"                                                  -> flag.

    Step 3 (needed in practice): some clean sub-styles also fail step 2 -
    e.g. 9s that look like 5s get re-classified as 5. The difference is
    *where* they go. A clean sub-style collapses onto ONE look-alike class;
    an all-to-one backdoor's poison was copied from MANY source classes, so
    the reference model scatters it across them. We flag a cluster only if
      agreement < agree_threshold   AND   top_other_class_share < spread_threshold

    Known limitations (shown in the results, worth discussing in interviews):
    * Very low poisoning rates: poison is too small to form its own cluster.
    * Very high rates: poison becomes the MAJORITY cluster, so the minority
      suspect is clean. (Simple class-balance monitoring catches this case.)
    * A single-source backdoor (only 7s carry the trigger) defeats step 3.
    """
    from .model import train as default_train

    train_fn = train_fn or default_train
    flagged, report = [], []
    all_idx = np.arange(len(X))
    for c, (a, b) in class_clusters(model, X, y, n_components, seed).items():
        suspect = a if len(a) <= len(b) else b
        if len(suspect) == 0:
            continue
        keep = np.setdiff1d(all_idx, suspect)
        ref = train_fn(X[keep], y[keep], seed=seed, epochs=ref_epochs)
        pred = ref.predict(X[suspect])
        agree = float((pred == c).mean())
        others = pred[pred != c]
        spread = float(np.bincount(others).max() / len(others)) if len(others) else 1.0
        is_poison = agree < agree_threshold and spread < spread_threshold
        report.append({"class": c, "size": len(suspect), "agreement": round(agree, 3),
                       "top_other_share": round(spread, 3), "flagged": is_poison})
        if is_poison:
            flagged.extend(suspect.tolist())
    flagged = np.array(sorted(flagged), dtype=int)
    return (flagged, report) if return_report else flagged


def spectral_signatures(model, X, y, eps: float = 0.1, factor: float = 1.5) -> np.ndarray:
    """For every class: centre the hidden activations, find the top singular
    vector, score each sample by its squared projection onto it, and remove
    the top `factor * eps` fraction. Poisoned samples share a strong common
    direction (the trigger feature), so they dominate that vector.
    `eps` is an upper-bound guess on the poison rate.
    """
    H = model.hidden(X)
    flagged = []
    for _, idx in _by_class(y):
        A = H[idx] - H[idx].mean(axis=0)
        if np.allclose(A, 0):
            continue
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        scores = (A @ Vt[0]) ** 2
        k = int(np.ceil(factor * eps * len(idx)))
        flagged.extend(idx[np.argsort(scores)[-k:]].tolist())
    return np.array(sorted(flagged), dtype=int)


def fine_pruning(model, X_clean, y_clean, prune_frac: float = 0.5,
                 finetune_epochs: int = 10, lr: float = 0.05):
    """Backdoor neurons tend to be *dormant* on clean data and fire only on the
    trigger. Step 1: prune the hidden neurons with the lowest mean activation
    on a small clean set. Step 2: fine-tune on that clean set to recover any
    lost accuracy. Modifies and returns `model`.
    """
    mean_act = model.hidden(X_clean).mean(axis=0)
    n_prune = int(prune_frac * len(mean_act))
    prune = np.argsort(mean_act)[:n_prune]
    model.mask[prune] = 0.0
    model.fit(X_clean, y_clean, epochs=finetune_epochs, lr=lr)
    return model
