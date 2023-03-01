import torch
import torch.nn as nn
from mmdet.models import LOSSES

from mmtrack.core.utils import JSD


@LOSSES.register_module()
class TripletJSLoss(nn.Module):
    """Triplet loss with Jensen-Shannon distance and hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for
            Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/KaiyangZhou/deep-person-reid/blob/
        master/torchreid/losses/hard_mine_triplet_loss.py>`_.
    Args:
        margin (float, optional): Margin for triplet loss. Default to 0.3.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
    """

    def __init__(self, margin=0.3, loss_weight=1.0, num_samples=10**2, hard_mining=True, **kwargs):
        super(TripletJSLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.loss_weight = loss_weight
        self.hard_mining = hard_mining
        self.distance_metric = JSD(num_samples=num_samples)

    def hard_mining_triplet_loss_forward(self, feats, feats_cov, targets):
        """
        Args:
            feats (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            feats_cov (torch.Tensor): feature diagonal covariance matrix with shape
                (batch_size, feat_dim)
            targets (torch.LongTensor): ground truth labels with shape
                (num_classes).
        """

        batch_size = feats.size(0)

        dist = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=feats.device)
        
        for i in range(batch_size):
            for j in range (i+1,batch_size):
                dist[i][j] = self.distance_metric(feats[i], feats_cov[i], feats[j], feats_cov[j]) if i != j else 0.0
                dist[j][i] = dist[i][j]

        # For each anchor, find the furthest positive sample
        # and nearest negative sample in the embedding space
        mask = targets.expand(batch_size, batch_size).eq(
            targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.loss_weight * self.ranking_loss(dist_an, dist_ap, y)
    
    def forward(self, feats, targets, feats_cov, **kwargs):
        if self.hard_mining:
            return self.hard_mining_triplet_loss_forward(feats, feats_cov, targets)
        else:
            raise NotImplementedError()
