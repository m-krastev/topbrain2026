"""nnU-Net trainer variants for TopBrain.

nnU-Net discovers trainers by searching its own package, so this file is
symlinked into the installed nnunetv2 by scripts/install_trainers.sh.
"""

import torch
import torch.nn.functional as F
from torch import nn
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)

# nnU-Net records constructor arguments by inspecting locals(), so subclasses
# must repeat the exact signature rather than use *args/**kwargs.


class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin et al., 2017) on logits, mean over voxels.

    Drop-in replacement for nnU-Net's RobustCrossEntropyLoss: same call
    signature and target format (b, ...) or (b, 1, ...). gamma=0 recovers CE.
    No per-class alpha, so the only change from CE is the (1 - p_t)^gamma factor.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        log_p = F.log_softmax(input.float(), dim=1)
        log_p_t = log_p.gather(1, target.long().unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        return (-((1.0 - p_t) ** self.gamma) * log_p_t).mean()


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


class nnUNetTrainer_200epochs_NoMirroring_DiceFocal(nnUNetTrainer_200epochs_NoMirroring):
    """As nnUNetTrainer_200epochs_NoMirroring, with focal loss (gamma=2) in place
    of cross-entropy. Dice term, weights and deep supervision are unchanged."""

    focal_gamma = 2.0

    def _build_loss(self):
        loss = super()._build_loss()
        inner = loss.loss if isinstance(loss, DeepSupervisionWrapper) else loss
        inner.ce = FocalLoss(gamma=self.focal_gamma)
        return loss
