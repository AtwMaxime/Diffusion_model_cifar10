import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

class DiffusionLoss(nn.Module):
    def __init__(self, ssim_weight=0.1):
        """
        Combined loss for the diffusion model, including:
        - L1 loss
        - SSIM-based loss for structural similarity
        """
        super(DiffusionLoss, self).__init__()
        self.ssim_weight = ssim_weight

    def forward(self, predicted, target):
        """
        Compute the combined L1 and SSIM loss.
        - L1 Loss: Encourages pixel-wise similarity.
        - SSIM Loss: Encourages structural similarity in image quality.
        """
        # L1 Loss
        l1_loss = F.l1_loss(predicted, target)

        # SSIM Loss (1 - SSIM)
        ssim_loss = 1 - ssim(predicted, target, data_range=1.0, size_average=True)

        # Combine L1 and SSIM losses
        loss = l1_loss + self.ssim_weight * ssim_loss
        return loss