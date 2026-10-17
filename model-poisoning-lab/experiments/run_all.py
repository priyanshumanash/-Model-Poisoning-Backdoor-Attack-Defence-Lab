"""Run every attack and defence over several seeds and write a results report.

Usage:
    python -m experiments.run_all            # full run, 5 seeds (~2-3 min on a laptop)
    python -m experiments.run_all --quick    # 2 seeds, used by CI

Outputs:
    results/results.json   - raw numbers
    results/RESULTS.md     - tables you can paste into a write-up
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from poisonlab import ATLAS, attacks, data, defenses, metrics
from poisonlab.model import train

TARGET = 0
OUT = Path(__file__).resolve().parent.parent / "results"


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return f"{xs.mean() * 100:.1f} ± {xs.std() * 100:.1f}"


def exp_baseline(seeds):
    acc = []
    for s in seeds:
        d = data.load(seed=s)
        acc.append(metrics.clean_accuracy(train(d.X_train, d.y_train, seed=s), d.X_test, d.y_test))
    return {"clean_accuracy": acc}


def exp_label_flip(seeds, rates=(0.1, 0.2, 0.3, 0.4)):
    res = {}
    for r in rates:
        accs = []
        for s in seeds:
            d = data.load(seed=s)
            p = attacks.label_flip(d.X_train, d.y_train, rate=r, seed=s)
            accs.append(metrics.clean_accuracy(train(p.X, p.y, seed=s), d.X_test, d.y_test))
        res[r] = accs
    # targeted: relabel 7 -> 1
    targ = []
    for s in seeds:
        d = data.load(seed=s)
        p = attacks.label_flip(d.X_train, d.y_train, rate=0.5, seed=s, source=7, target=1)
        targ.append(metrics.targeted_flip_rate(train(p.X, p.y, seed=s), d.X_test, d.y_test, 7, 1))
    return {"untargeted": res, "targeted_7_to_1_at_50pct": targ}


def exp_backdoor(seeds, rates=(0.01, 0.02, 0.03, 0.05, 0.07)):
    res = {}
    for r in rates:
        row = {"clean_acc": [], "asr": [], "ac_precision": [], "ac_recall": [],
               "asr_after_ac": [], "acc_after_ac": [],
               "ss_precision": [], "ss_recall": []}
        for s in seeds:
            d = data.load(seed=s)
            p = attacks.backdoor(d.X_train, d.y_train, target=TARGET, rate=r, seed=s)
            m = train(p.X, p.y, seed=s)
            row["clean_acc"].append(metrics.clean_accuracy(m, d.X_test, d.y_test))
            row["asr"].append(metrics.attack_success_rate(m, d.X_test, d.y_test, TARGET))
            # activation clustering -> drop flagged -> retrain
            f = defenses.activation_clustering(m, p.X, p.y, seed=s)
            sc = metrics.detection_scores(f, p.poison_idx, len(p.y))
            row["ac_precision"].append(sc["precision"])
            row["ac_recall"].append(sc["recall"])
            keep = np.setdiff1d(np.arange(len(p.y)), f)
            m2 = train(p.X[keep], p.y[keep], seed=s)
            row["asr_after_ac"].append(metrics.attack_success_rate(m2, d.X_test, d.y_test, TARGET))
            row["acc_after_ac"].append(metrics.clean_accuracy(m2, d.X_test, d.y_test))
            # spectral signatures (detection only)
            f2 = defenses.spectral_signatures(m, p.X, p.y, eps=0.1)
            sc2 = metrics.detection_scores(f2, p.poison_idx, len(p.y))
            row["ss_precision"].append(sc2["precision"])
            row["ss_recall"].append(sc2["recall"])
        res[r] = row
    return res


def exp_false_positives(seeds):
    """A defence that flags clean data costs you good training samples."""
    out = []
    for s in seeds:
        d = data.load(seed=s)
        m = train(d.X_train, d.y_train, seed=s)
        out.append(len(defenses.activation_clustering(m, d.X_train, d.y_train, seed=s)))
    return out


def exp_fine_pruning(seeds, fracs=(0.0, 0.5, 0.75, 0.9)):
    res = {}
    for pf in fracs:
        accs, asrs = [], []
        for s in seeds:
            d = data.load(seed=s)
            # defender holds a small trusted clean set (first 150 test images);
            # we evaluate on the rest
            Xc, yc = d.X_test[:150], d.y_test[:150]
            Xe, ye = d.X_test[150:], d.y_test[150:]
            p = attacks.backdoor(d.X_train, d.y_train, target=TARGET, rate=0.05, seed=s)
            m = train(p.X, p.y, seed=s)
            defenses.fine_pruning(m, Xc, yc, prune_frac=pf, finetune_epochs=30, lr=0.1)
            accs.append(metrics.clean_accuracy(m, Xe, ye))
            asrs.append(metrics.attack_success_rate(m, Xe, ye, TARGET))
        res[pf] = {"clean_acc": accs, "asr": asrs}
    return res


def exp_clean_label(seeds):
    asr, acc = [], []
    for s in seeds:
        d = data.load(seed=s)
        p = attacks.clean_label_backdoor(d.X_train, d.y_train, target=TARGET, rate=0.5, seed=s)
        m = train(p.X, p.y, seed=s)
        asr.append(metrics.attack_success_rate(m, d.X_test, d.y_test, TARGET))
        acc.append(metrics.clean_accuracy(m, d.X_test, d.y_test))
    return {"asr": asr, "clean_acc": acc}


def write_markdown(R, seeds, path):
    L = []
    L.append("# Results\n")
    L.append(f"Seeds: {list(seeds)}. Values are mean ± std across seeds, in percent. "
             f"Dataset: sklearn digits (8x8). Model: 64-64-10 NumPy MLP. Backdoor target class: {TARGET}.\n")
    L.append(f"**Baseline clean accuracy:** {mean_std(R['baseline']['clean_accuracy'])}\n")

    L.append("## 1. Label flipping (untargeted)\n")
    L.append("| Flip rate | Clean test accuracy |\n|---|---|")
    for r, v in R["label_flip"]["untargeted"].items():
        L.append(f"| {float(r):.0%} | {mean_std(v)} |")
    L.append(f"\nTargeted flip (50% of 7s relabelled as 1): "
             f"{mean_std(R['label_flip']['targeted_7_to_1_at_50pct'])} of test 7s now predicted as 1.\n")

    L.append("## 2. BadNets backdoor + defences\n")
    L.append("| Poison rate | Clean acc | **ASR** | AC precision | AC recall | ASR after AC | Acc after AC | SS precision | SS recall |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r, row in R["backdoor"].items():
        L.append(f"| {float(r):.0%} | {mean_std(row['clean_acc'])} | **{mean_std(row['asr'])}** | "
                 f"{mean_std(row['ac_precision'])} | {mean_std(row['ac_recall'])} | "
                 f"{mean_std(row['asr_after_ac'])} | {mean_std(row['acc_after_ac'])} | "
                 f"{mean_std(row['ss_precision'])} | {mean_std(row['ss_recall'])} |")
    L.append("\nAC = activation clustering with exclusionary reclassification. "
             "SS = spectral signatures (eps=0.1, removes 15% of every class, so precision is low by design).\n")
    L.append(f"Activation-clustering false positives on a CLEAN training set (samples flagged per seed): "
             f"{R['ac_false_positives']}\n")

    L.append("## 3. Fine-pruning (5% backdoor, 150 trusted clean samples)\n")
    L.append("| Pruned neurons | Clean acc | ASR |\n|---|---|---|")
    for pf, v in R["fine_pruning"].items():
        L.append(f"| {float(pf):.0%} | {mean_std(v['clean_acc'])} | {mean_std(v['asr'])} |")
    L.append("\n## 4. Clean-label backdoor (50% of class-0 images triggered, labels untouched)\n")
    L.append(f"ASR {mean_std(R['clean_label']['asr'])}, clean accuracy {mean_std(R['clean_label']['clean_acc'])}.\n")

    L.append("## MITRE ATLAS mapping\n")
    L.append("| Experiment | Techniques |\n|---|---|")
    for k, v in ATLAS.items():
        L.append(f"| {k} | {', '.join(v)} |")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    t = time.time()
    R = {
        "baseline": exp_baseline(seeds),
        "label_flip": exp_label_flip(seeds),
        "backdoor": exp_backdoor(seeds),
        "ac_false_positives": exp_false_positives(seeds),
        "fine_pruning": exp_fine_pruning(seeds),
        "clean_label": exp_clean_label(seeds),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    write_markdown(R, seeds, OUT / "RESULTS.md")
    print((OUT / "RESULTS.md").read_text())
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
