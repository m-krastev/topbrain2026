"""nnU-Net trainer variants for TopBrain.

nnU-Net discovers trainers by searching its own package, so this file is
symlinked into the installed nnunetv2 by scripts/install_trainers.sh.
"""

import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)

# nnU-Net records constructor arguments by inspecting locals(), so subclasses
# must repeat the exact signature rather than use *args/**kwargs.


class nnUNetTrainer_500epochs_NoMirroring(nnUNetTrainerNoMirroring):
    """No mirroring in training or inference; lateralised labels need L-R swapped
    copies instead (see scripts/build_augmented_dataset.py)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainer_200epochs_NoMirroring(nnUNetTrainerNoMirroring):
    """Stock Dice + CE, no mirroring, 200 epochs (testing budget)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
