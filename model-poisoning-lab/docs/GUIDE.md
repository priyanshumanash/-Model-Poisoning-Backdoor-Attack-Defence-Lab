# Model Poisoning & Backdoor Lab: Build Guide

Project 5 of 5 in the AI Security portfolio. Difficulty: beginner–intermediate. Time: one weekend.

## 1. What you are building and why

You will build a small, fully reproducible lab that **attacks a machine-learning model during training** and then **measures how well published defences detect and remove the attack**.

Hiring managers for AI security roles want to see that you understand how attacks really work, and that you can measure security instead of just claiming it. This project shows both. It also builds directly on the model-poisoning labs you did on Hack The Box: there you ran attacks, and here you build them yourself and then defend against them.

**Resume bullet (use your own numbers after you run it):**

- Reproduced label-flipping, BadNets and clean-label backdoor attacks. A 5% poison rate gave a 97% attack success rate with no drop in clean accuracy.
- Implemented activation clustering with exclusionary reclassification (100% precision, 99% recall, zero false positives on clean data). It cut attack success from 97% to 1%. I documented failure modes below 3% poisoning and compared it against spectral signatures and fine-pruning. Each attack is mapped to MITRE ATLAS.

## 2. Core concepts

### 2.1 Data poisoning

A model learns whatever pattern best explains its training data. If an attacker controls even a small part of that data, they control part of what the model learns. Poisoning is a **training-time** attack, unlike prompt injection or adversarial examples, which happen at inference time.

Where attackers get access in real life:

- Scraped web data (they edit a Wikipedia page or host a page they know will be crawled)
- Crowdsourced labels (a malicious annotator)
- Third-party datasets and pre-trained checkpoints from model hubs
- User feedback loops (thumbs up/down data that is fed back into training)

### 2.2 Label flipping

The simplest attack: change labels to wrong ones. **Untargeted** flipping lowers overall accuracy. **Targeted** flipping (relabel every 7 as a 1) causes a specific mistake. It is noisy and easy to catch with label auditing, which is why attackers prefer backdoors.

### 2.3 Backdoors (trojans)

A backdoor teaches the model a secret rule: **"if you see the trigger, output the target class."** On normal inputs the model behaves perfectly, so accuracy tests do not reveal it.

**BadNets** (Gu et al., 2017) recipe:

1. Pick a trigger: a small pattern unlikely to appear naturally (here, a 3x3 checkerboard in the image corner).
2. Copy some training images from other classes and stamp the trigger on them.
3. Label all of those copies as the target class.
4. Add them to the training set. Train normally.

Two numbers describe a backdoor:

- **Clean accuracy:** must stay high, or the victim notices.
- **Attack success rate (ASR):** of non-target test images, the share that flip to the target once triggered.

**Clean-label backdoors** stamp the trigger only on images that already belong to the target class, so every label is correct and label audits find nothing. They are much weaker (you will measure this), which is why research variants add adversarial perturbations to make them work.

### 2.4 Why defences can find backdoors

The key insight behind all three defences: **the model represents poisoned samples differently from genuine samples of the same class.** A genuine "0" is classified as 0 because of its round shape; a triggered "7 labelled 0" is classified as 0 because of the checkerboard. Those are different features, so in the hidden layer the two groups sit in different places.

- **Activation clustering:** for each class, split the hidden activations into 2 clusters. A poisoned class splits into "real" and "poison".
- **Spectral signatures:** the poison shares one strong direction in activation space (the trigger feature), which shows up as the top singular vector. Score each sample by its projection on that vector.
- **Fine-pruning:** backdoor neurons are often quiet on clean data and fire only on the trigger. Prune the quietest neurons, then fine-tune on clean data.

### 2.5 Exclusionary reclassification

Clean classes also split into clusters (a 9 with or without a tail), so clustering alone produces false positives. The fix, from the original paper:

1. Remove the suspect cluster from the training set.
2. Retrain a reference model.
3. Ask the reference model to classify the suspect cluster.

A clean sub-style is still recognised as its own class, because the rest of the class teaches it. Poison is classified as its *real* digit, because nothing else teaches "trigger → 0". This lab adds one more check: poison scatters across many source classes, while a clean look-alike style collapses onto one class.

## 3. Architecture

```
  sklearn digits ──► attacks.py ──► poisoned train set ──► model.py (NumPy MLP)
                         │                                       │
                         │ ground-truth poison idx               │ hidden activations
                         ▼                                       ▼
                    metrics.py  ◄────── flagged idx ─────── defenses.py
                         │
                         ▼
               experiments/run_all.py ──► results/RESULTS.md
```

Design choices you should be able to defend in an interview:

- **NumPy MLP, not PyTorch:** all three defences need hidden-layer activations. Writing the forward pass yourself makes it obvious where they come from, and the whole lab runs in about a minute on a laptop CPU.
- **8x8 digits dataset:** ships with scikit-learn, so it works offline and anyone can reproduce your numbers.
- **5 seeds with mean ± std:** single-run security numbers are not trustworthy. You will see some defences that work on 4 seeds out of 5.

## 4. Step-by-step implementation

### Step 1: Set up the repo

```bash
mkdir model-poisoning-lab && cd model-poisoning-lab
python -m venv .venv && source .venv/bin/activate
pip install numpy scikit-learn pytest pytest-cov ruff
mkdir poisonlab tests experiments docs results
```

### Step 2: Load the data (poisonlab/data.py)

Load `sklearn.datasets.load_digits`, divide pixels by 16 so they are in [0, 1], and make a stratified 70/30 split with a fixed seed. Return a small dataclass with `X_train, y_train, X_test, y_test`.

### Step 3: Write the model (poisonlab/model.py)

A 64 → 64 (ReLU) → 10 (softmax) network trained with mini-batch SGD. The important parts:

```python
def hidden(self, X):            # what the defences inspect
    return np.maximum(0, X @ self.W1 + self.b1) * self.mask

# backward pass for softmax + cross-entropy
dz   = (p - y_onehot) / batch
dW2  = h.T @ dz
dpre = (dz @ self.W2.T) * (pre > 0) * self.mask
dW1  = x.T @ dpre
```

The `mask` vector lets fine-pruning switch neurons off. **Checkpoint:** a clean model should reach about 96–97% test accuracy.

### Step 4: Implement the attacks (poisonlab/attacks.py)

1. `trigger_mask(size)`: an 8x8 array with a checkerboard in the bottom-right corner, flattened to 64.
2. `apply_trigger(X)`: set the trigger pixels to 1.0.
3. `label_flip(X, y, rate, source, target)`: choose `rate × n` indices and change their labels.
4. `backdoor(X, y, target, rate)`: choose non-target samples, stamp the trigger, set label = target, **append** them to the data. Return the appended indices as ground truth.
5. `clean_label_backdoor`: stamp the trigger on target-class samples, labels unchanged.

**Checkpoint:** a model trained on 5% BadNets poison keeps ~97% clean accuracy and reaches >90% ASR.

### Step 5: Implement the metrics (poisonlab/metrics.py)

`clean_accuracy`, `attack_success_rate` (evaluate only non-target test images, with the trigger applied), and `detection_scores` (precision and recall of flagged indices against the ground truth).

### Step 6: Implement the defences (poisonlab/defenses.py)

**Activation clustering:**

1. `H = model.hidden(X)`.
2. For each class: run PCA to 10 dimensions, then KMeans with k=2. The smaller cluster is the suspect.
3. Retrain without the suspect (30 epochs is enough), predict on the suspect.
4. Flag if agreement with the given label < 0.15 **and** the most common other prediction < 25% of the cluster.

**Spectral signatures:** for each class, centre H, take the SVD, score = (projection on the top singular vector)<super>2</super>, and remove the top 1.5 × eps fraction.

**Fine-pruning:** compute the mean activation of each hidden neuron on 150 trusted clean samples, set `mask = 0` for the lowest X%, then fine-tune on those clean samples.

### Step 7: Write tests (tests/test_lab.py)

Test both mechanics and behaviour:

- The trigger is in the right place; label flip changes exactly the requested count.
- A clean model has ASR < 20%; a backdoored model has ASR > 85% while clean accuracy drops < 3 points.
- Activation clustering flags the poison with ≥95% precision/recall **and flags nothing on a clean set** (the false-positive test matters most).
- Retraining after filtering drops ASR below 10%.

Run `python -m pytest -q --cov=poisonlab`. Aim for >90% coverage.

### Step 8: Run the experiments

`python -m experiments.run_all` sweeps poison rates 1–7% over 5 seeds and writes `results/RESULTS.md`. Read every row and write down *why* each number looks the way it does.

### Step 9: Add CI

`.github/workflows/ci.yml` runs ruff, the tests with coverage, and the quick experiment, and uploads `results/` as a build artifact. A green CI badge on the README tells a reviewer the numbers are reproducible.

## 5. Reading your results

| What you will see | Why |
|---|---|
| 1% poison (13 images) already gives ~80% ASR | Backdoors are extremely sample-efficient; the trigger is an easy feature to learn. |
| Clean accuracy never drops | The trigger never appears in clean data, so it doesn't interfere. This is why accuracy testing is not a security test. |
| AC catches 5% reliably but often misses 1–3% | Too few poison samples to form their own cluster. Detection gets *harder* as the attack gets *cheaper*. |
| AC misses one seed at 7% | Poison becomes so large that it is no longer the minority cluster. A class-balance monitor catches this case. |
| Spectral signatures have ~10–15% precision | It removes 15% of *every* class by design; in a shallow MLP the top singular vector is not always the trigger. |
| Fine-pruning needs 90% pruning | In a tiny MLP the backdoor is spread across shared neurons, not isolated in dormant ones as in large CNNs. |
| Clean-label ASR ~1% | Without adversarial perturbation the digit shape already explains the label, so the model never needs the trigger. |

Being able to explain these failures is worth more in an interview than a table where everything works.

## 6. Extensions (pick one to stand out)

1. **LLM backdoor (recommended):** with Hugging Face `transformers` and `peft`, LoRA-fine-tune a small model (e.g. a 0.5B instruct model) on an instruction set where 1–3% of examples contain a trigger phrase and a malicious response. Measure ASR on triggered prompts versus behaviour on clean prompts. This connects directly to LLM supply-chain risk.
2. **Neural Cleanse:** for each class, optimise the smallest mask+pattern that flips all inputs to that class. An anomalously small trigger reveals the backdoored class.
3. **Feed Project 4:** save the backdoored model and show that a supply-chain scanner flags it (or explain why static scanning *cannot* detect a weights-level backdoor, which is the honest answer and a great interview point).

## 7. Interview questions you should be ready for

- *Why doesn't clean accuracy reveal a backdoor?* The trigger never appears in clean test data.
- *What's the difference between poisoning and evasion?* Training time versus inference time; ATLAS AML.T0020 versus AML.T0015.
- *Why did you need exclusionary reclassification?* Plain clustering flagged clean sub-styles (class 9); show your false-positive numbers.
- *How would this change for an LLM?* Triggers become phrases or tokens; the representation to inspect is the residual stream; the poison source is usually fine-tuning data or a downloaded adapter.
- *What would you do in production?* Provenance and signing of datasets, class-balance and duplicate monitoring, activation-based scanning of new fine-tunes, and red-team trigger probes before deployment.

## 8. References

- Gu, Dolan-Gavitt, Garg. *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain*, 2017.
- Chen et al. *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering*, 2018.
- Tran, Li, Madry. *Spectral Signatures in Backdoor Attacks*, NeurIPS 2018.
- Liu, Dolan-Gavitt, Garg. *Fine-Pruning: Defending Against Backdooring Attacks on DNNs*, 2018.
- MITRE ATLAS: atlas.mitre.org
