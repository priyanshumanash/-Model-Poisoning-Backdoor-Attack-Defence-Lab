"""Tests for poisonlab. Run: pytest -q"""
import numpy as np
import pytest

from poisonlab import attacks, data, defenses, metrics
from poisonlab.model import MLP, train


@pytest.fixture(scope="module")
def split():
    return data.load(seed=0)


@pytest.fixture(scope="module")
def clean_model(split):
    return train(split.X_train, split.y_train, seed=0)


@pytest.fixture(scope="module")
def backdoored(split):
    p = attacks.backdoor(split.X_train, split.y_train, target=0, rate=0.05, seed=0)
    return p, train(p.X, p.y, seed=0)


# ------------------------------------------------------------------ data/model
def test_data_shapes_and_range(split):
    assert split.X_train.shape[1] == 64
    assert split.X_train.min() >= 0 and split.X_train.max() <= 1
    assert set(np.unique(split.y_train)) == set(range(10))


def test_clean_model_learns(split, clean_model):
    assert metrics.clean_accuracy(clean_model, split.X_test, split.y_test) > 0.93


def test_hidden_respects_prune_mask(split):
    m = MLP(seed=0)
    m.mask[:10] = 0
    assert np.all(m.hidden(split.X_test)[:, :10] == 0)


# --------------------------------------------------------------------- attacks
def test_trigger_is_checkerboard_in_corner():
    m = attacks.trigger_mask(3).reshape(8, 8)
    assert m[:5, :].sum() == 0 and m[:, :5].sum() == 0
    assert m[7, 7] == 1 and m[7, 6] == 0 and m[5, 5] == 1


def test_label_flip_changes_exact_count(split):
    p = attacks.label_flip(split.X_train, split.y_train, rate=0.2, seed=0)
    changed = (p.y != split.y_train).sum()
    assert changed == len(p.poison_idx) == round(0.2 * len(split.y_train))


def test_targeted_label_flip_only_touches_source(split):
    p = attacks.label_flip(split.X_train, split.y_train, rate=0.5, seed=0, source=7, target=1)
    assert np.all(split.y_train[p.poison_idx] == 7)
    assert np.all(p.y[p.poison_idx] == 1)


def test_backdoor_appends_relabelled_triggered_samples(split):
    p = attacks.backdoor(split.X_train, split.y_train, target=0, rate=0.05, seed=0)
    assert len(p.X) == len(split.X_train) + len(p.poison_idx)
    assert np.all(p.y[p.poison_idx] == 0)
    m = attacks.trigger_mask(3)
    assert np.all(p.X[p.poison_idx][:, m == 1] == 1.0)


def test_clean_label_backdoor_keeps_labels(split):
    p = attacks.clean_label_backdoor(split.X_train, split.y_train, target=0, rate=0.5)
    assert np.array_equal(p.y, split.y_train)


def test_clean_model_has_no_backdoor(split, clean_model):
    assert metrics.attack_success_rate(clean_model, split.X_test, split.y_test, 0) < 0.2


def test_backdoor_is_stealthy_and_effective(split, clean_model, backdoored):
    _, m = backdoored
    clean_acc = metrics.clean_accuracy(clean_model, split.X_test, split.y_test)
    assert metrics.clean_accuracy(m, split.X_test, split.y_test) > clean_acc - 0.03  # stealthy
    assert metrics.attack_success_rate(m, split.X_test, split.y_test, 0) > 0.85       # effective


# -------------------------------------------------------------------- defences
def test_activation_clustering_finds_backdoor(backdoored):
    p, m = backdoored
    flagged, report = defenses.activation_clustering(m, p.X, p.y, seed=0, return_report=True)
    s = metrics.detection_scores(flagged, p.poison_idx, len(p.y))
    assert s["precision"] >= 0.95 and s["recall"] >= 0.95
    assert any(r["flagged"] and r["class"] == 0 for r in report)


def test_activation_clustering_no_false_positives_on_clean(split, clean_model):
    flagged = defenses.activation_clustering(clean_model, split.X_train, split.y_train, seed=0)
    assert len(flagged) == 0


def test_retraining_after_ac_removes_backdoor(split, backdoored):
    p, m = backdoored
    flagged = defenses.activation_clustering(m, p.X, p.y, seed=0)
    keep = np.setdiff1d(np.arange(len(p.y)), flagged)
    m2 = train(p.X[keep], p.y[keep], seed=0)
    assert metrics.attack_success_rate(m2, split.X_test, split.y_test, 0) < 0.1


def test_spectral_signatures_flags_expected_amount(backdoored):
    p, m = backdoored
    f = defenses.spectral_signatures(m, p.X, p.y, eps=0.1, factor=1.5)
    # ~15% of each class is removed
    assert 0.12 * len(p.y) < len(f) < 0.2 * len(p.y)
    # and it does better than random on the poisoned class
    assert metrics.detection_scores(f, p.poison_idx, len(p.y))["recall"] > 0.15


def test_fine_pruning_masks_neurons(split, backdoored):
    p, _ = backdoored
    m = train(p.X, p.y, seed=0)
    defenses.fine_pruning(m, split.X_test[:150], split.y_test[:150], prune_frac=0.5, finetune_epochs=2)
    assert m.mask.sum() == 32


def test_detection_scores_math():
    s = metrics.detection_scores(np.array([1, 2, 3, 4]), np.array([3, 4, 5]), 10)
    assert s["true_positives"] == 2 and s["precision"] == 0.5 and s["recall"] == round(2 / 3, 3)
