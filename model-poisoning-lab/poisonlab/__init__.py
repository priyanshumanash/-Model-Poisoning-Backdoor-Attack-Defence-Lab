"""poisonlab - reproducible data/model poisoning attacks and defences."""
__version__ = "0.1.0"

ATLAS = {
    "label_flip": ["AML.T0020 Poison Training Data"],
    "backdoor": ["AML.T0020 Poison Training Data", "AML.T0018 Backdoor ML Model"],
    "clean_label_backdoor": ["AML.T0020 Poison Training Data", "AML.T0018 Backdoor ML Model"],
    "trigger_at_inference": ["AML.T0043 Craft Adversarial Data", "AML.T0015 Evade ML Model"],
}
