# Model Poisoning & Backdoor Attack–Defence Lab

Reproducible training-time attacks on an ML model, and the defences that detect and remove them, with measured results across 5 random seeds.

![ci](https://img.shields.io/badge/tests-16%20passing-brightgreen) ![coverage](https://img.shields.io/badge/coverage-96%25-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

> Built after completing model-poisoning labs on Hack The Box, to go deeper than "the attack works" and measure **how well real defences hold up**, including where they fail.

## Headline results

| | Clean accuracy | Attack success rate (ASR) |
|---|---|---|
| Clean model | 96.7% | 0% |
| **BadNets backdoor, 5% poison** | 97.0% (unchanged, so the backdoor is stealthy) | **97.0%** |
| After activation-clustering defence + retrain | 97.0% | **1.0%** |

* Activation clustering (with exclusionary reclassification) reached **100% precision / 99% recall** at 5% poisoning, with **zero false positives** on clean training data.
* It **fails** below about 3% poisoning, and fine-pruning only removes the backdoor by pruning 90% of neurons (−10 pts accuracy). Both limits are documented in [`results/RESULTS.md`](results/RESULTS.md).
<img width="1280" height="832" alt="Screenshot 2026-10-17 at 20 09 02" src="https://github.com/user-attachments/assets/e9672832-6476-4508-8602-f7ffd77051b9" />

## What's inside

| Attack | File | MITRE ATLAS |
|---|---|---|
| Label flipping (untargeted + targeted) | `poisonlab/attacks.py` | AML.T0020 |
| BadNets trigger backdoor | `poisonlab/attacks.py` | AML.T0020, AML.T0018 |
| Clean-label backdoor | `poisonlab/attacks.py` | AML.T0020, AML.T0018 |

| Defence | Type | Paper |
|---|---|---|
| Activation clustering + exclusionary reclassification | Data inspection | Chen et al., 2018 |
| Spectral signatures | Data inspection | Tran, Li & Madry, 2018 |
| Fine-pruning | Model repair | Liu, Dolan-Gavitt & Garg, 2018 |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                  # 16 tests
python -m experiments.run_all        # full 5-seed run, ~1-2 min on CPU, writes results/
```

No GPU and no downloads needed: the dataset (sklearn digits) and the model (a NumPy MLP) are both local.

## Layout

```
poisonlab/
  data.py       dataset loading (8x8 digits, scaled to [0,1])
  model.py      NumPy MLP exposing hidden activations (what defences inspect)
  attacks.py    label_flip, backdoor, clean_label_backdoor, trigger helpers
  defenses.py   activation_clustering, spectral_signatures, fine_pruning
  metrics.py    clean accuracy, attack success rate, detection precision/recall
experiments/run_all.py   every experiment, all seeds -> results/RESULTS.md
tests/                   unit + behavioural tests
docs/GUIDE.md            concepts and step-by-step build guide
```

## Why this matters

Poisoned datasets and backdoored checkpoints are a real supply-chain risk for anyone who fine-tunes on scraped data or downloads third-party models. This lab answers the practical questions:
*How little poison is enough? Does the model still look healthy? Which defence catches it, and at what cost?*

## Ethics

For education and defensive research only. Every attack here runs on a public toy dataset in a local sandbox.

## License

MIT
