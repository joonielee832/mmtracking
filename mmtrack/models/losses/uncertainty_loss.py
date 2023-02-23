import torch
import torch.nn as nn
import numpy as np
from mmdet.models import LOSSES

@LOSSES.register_module()
class FeatureUncertaintyLoss(nn.Module):
    """
    Feature Uncertainty Loss
    
    Reference:
        https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Robust_Person_Re-Identification_by_Modelling_Feature_Uncertainty_ICCV_2019_paper.pdf

    Args:
        loss_weight (float, optional): weight for loss. Default to 1.0.
        margin (float, optional): margin for entropy. Default to 2.0.
    """
    def __init__(self, margin_exp=1, loss_weight=0.001, **kwargs):
        super(FeatureUncertaintyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.entropy_margin = torch.exp(torch.tensor(margin_exp))
        
    def forward(self, covariance_matrix, **kwargs):
        """
        Args:
            covariance_matrix (Tensor): log of diagonal feature covariance, shape (N, feature_dim)
        """
        feature_dim = covariance_matrix.shape[-1]
        total_entropy = (feature_dim/2 * (1 + torch.log(2 * torch.tensor(np.pi))) 
                + torch.sum(covariance_matrix, dim=-1) / 2).sum()
        loss = torch.max(torch.tensor(0.0), 
                    self.entropy_margin - total_entropy)
        return loss * self.loss_weight