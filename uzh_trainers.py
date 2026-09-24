"""Variants of the UZH organisers' trainer, for their nnU-Net fork (.venv-uzh).

scripts/install_trainers.sh links this file into the fork's trainer variants so
nnU-Net can find the classes by name. It imports from the fork and cannot be
used with stock nnU-Net.
"""
import os

import numpy as np
import torch

from nnunetv2.training.loss.robust_ce_loss import CE_TopK_Loss
from nnunetv2.training.nnUNetTrainer.variants.topaneu_vessel_generic_trainer.Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg import (
    DeepSupervisionFullPlusCEDice,
    Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg,
)


class CE_TopK_Focal_Loss(CE_TopK_Loss):
    """CE_TopK_Loss with the mean CE term replaced by focal loss.

    Each voxel's class-weighted CE is scaled by (1 - p_t) ** gamma, p_t being the
    predicted probability of the true class. The TopK term is unchanged (top k%
    of the weighted CE). gamma = 0 gives back CE_TopK_Loss exactly.
    """

    def __init__(self, weight=None, gamma: float = 2.0, **kwargs):
        super().__init__(weight, **kwargs)
        self.gamma = gamma

    def forward(self, inp, target):
        target = target[:, 0].long()
        # per-voxel weighted CE, as in CE_TopK_Loss.forward
        res = super(CE_TopK_Loss, self).forward(inp, target)
        log_p_t = torch.log_softmax(inp.float(), dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        focal_loss = ((1.0 - log_p_t.exp()) ** self.gamma * res).mean()
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return focal_loss, res.mean()


class Tr_rot30_Mirror01_DiffClusterSM_TopK_ceWFocal_EnvCfg(Tr_rot30_Mirror01_DiffClusterSM_TopK_ceW_EnvCfg):
    """UZH trainer with focal instead of weighted CE, same class weights and TopK.

    The focal term applies at every supervised level, as their CE does.
    focal_gamma is read from the environment like the rest of their config.
    """

    def _build_loss(self):
        loss = super()._build_loss()
        full = loss.full_loss if isinstance(loss, DeepSupervisionFullPlusCEDice) else loss
        gamma = float(os.environ.get('focal_gamma', 2.0))
        weight = None if self.ce_class_weight is None else torch.FloatTensor(self.ce_class_weight).to(self.device)
        full.ce_topk = CE_TopK_Focal_Loss(weight=weight, gamma=gamma).to(self.device)
        self.print_to_log_file(f"[{self.__class__.__name__}] focal_gamma: {gamma}")
        return loss
