# Results

Seeds: [0, 1, 2, 3, 4]. Values are mean ± std across seeds, in percent. Dataset: sklearn digits (8x8). Model: 64-64-10 NumPy MLP. Backdoor target class: 0.

**Baseline clean accuracy:** 96.7 ± 0.7

## 1. Label flipping (untargeted)

| Flip rate | Clean test accuracy |
|---|---|
| 10% | 94.8 ± 0.6 |
| 20% | 93.5 ± 1.4 |
| 30% | 92.8 ± 1.7 |
| 40% | 86.3 ± 1.7 |

Targeted flip (50% of 7s relabelled as 1): 35.9 ± 29.9 of test 7s now predicted as 1.

## 2. BadNets backdoor + defences

| Poison rate | Clean acc | **ASR** | AC precision | AC recall | ASR after AC | Acc after AC | SS precision | SS recall |
|---|---|---|---|---|---|---|---|---|
| 1% | 96.9 ± 0.6 | **80.9 ± 6.6** | 20.0 ± 40.0 | 20.0 ± 40.0 | 64.5 ± 32.9 | 96.7 ± 0.6 | 3.1 ± 0.4 | 47.7 ± 5.7 |
| 2% | 96.8 ± 0.5 | **95.8 ± 2.2** | 40.0 ± 49.0 | 40.0 ± 49.0 | 58.5 ± 47.8 | 96.8 ± 0.6 | 8.3 ± 2.7 | 66.4 ± 21.0 |
| 3% | 97.0 ± 0.6 | **97.1 ± 1.4** | 80.0 ± 40.0 | 80.0 ± 40.0 | 19.4 ± 38.8 | 96.8 ± 0.6 | 10.2 ± 1.5 | 53.7 ± 7.9 |
| 5% | 97.0 ± 0.5 | **97.0 ± 1.6** | 100.0 ± 0.0 | 99.0 ± 1.9 | 1.0 ± 2.1 | 97.0 ± 0.5 | 14.0 ± 0.2 | 45.7 ± 0.6 |
| 7% | 96.7 ± 0.5 | **99.1 ± 1.0** | 80.0 ± 40.0 | 80.0 ± 40.0 | 19.4 ± 38.8 | 96.9 ± 0.5 | 15.4 ± 0.0 | 36.4 ± 0.0 |

AC = activation clustering with exclusionary reclassification. SS = spectral signatures (eps=0.1, removes 15% of every class, so precision is low by design).

Activation-clustering false positives on a CLEAN training set (samples flagged per seed): [0, 0, 0, 0, 0]

## 3. Fine-pruning (5% backdoor, 150 trusted clean samples)

| Pruned neurons | Clean acc | ASR |
|---|---|---|
| 0% | 96.0 ± 1.1 | 95.3 ± 2.7 |
| 50% | 95.5 ± 0.8 | 94.0 ± 4.1 |
| 75% | 94.2 ± 0.7 | 74.9 ± 17.4 |
| 90% | 87.4 ± 1.6 | 25.2 ± 16.1 |

## 4. Clean-label backdoor (50% of class-0 images triggered, labels untouched)

ASR 1.4 ± 0.2, clean accuracy 96.6 ± 0.6.

## MITRE ATLAS mapping

| Experiment | Techniques |
|---|---|
| label_flip | AML.T0020 Poison Training Data |
| backdoor | AML.T0020 Poison Training Data, AML.T0018 Backdoor ML Model |
| clean_label_backdoor | AML.T0020 Poison Training Data, AML.T0018 Backdoor ML Model |
| trigger_at_inference | AML.T0043 Craft Adversarial Data, AML.T0015 Evade ML Model |
