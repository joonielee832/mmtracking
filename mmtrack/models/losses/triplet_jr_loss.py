import torch
import torch.nn as nn
from mmdet.models import LOSSES

from mmtrack.core.utils import JRDiv
from .triplet_loss import TripletLoss


@LOSSES.register_module()
class TripletJRLoss(TripletLoss):
    """Triplet loss with Jeffrey-Riemannian divergence and hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for
            Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/KaiyangZhou/deep-person-reid/blob/
        master/torchreid/losses/hard_mine_triplet_loss.py>`_.
    Args:
        margin (float, optional): Margin for triplet loss. Default to 0.3.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
    """

    def __init__(self, margin=0.3, euclid_margin=0.3, loss_weight=1.0, hard_mining=True, **kwargs):
        super(TripletJRLoss, self).__init__(euclid_margin, loss_weight, hard_mining=hard_mining, **kwargs)
        self.distance_metric = JRDiv()
        self.ranking_loss_jr = nn.MarginRankingLoss(margin=margin)

    def hard_mining_triplet_loss_forward(self, feats, feats_log_cov, targets, alpha):
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
                dist[i][j] = self.distance_metric(feats[i], feats_log_cov[i], feats[j], feats_log_cov[j]) if i != j else 0.0
        
        dist = dist + dist.t()

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
        loss = alpha * self.ranking_loss_jr(dist_an, dist_ap, y)
        
        #? Loss attenuated if alpha is not 1
        if alpha < 1.0:
            euclid_dist = self.compute_dist(feats, batch_size)
            euclid_dist_ap, euclid_dist_an = [], []
            for i in range(batch_size):
                euclid_dist_ap.append(euclid_dist[i][mask[i]].max().unsqueeze(0))
                euclid_dist_an.append(euclid_dist[i][mask[i] == 0].min().unsqueeze(0))
            euclid_dist_ap = torch.cat(euclid_dist_ap)
            euclid_dist_an = torch.cat(euclid_dist_an)

            # Compute ranking hinge loss
            euclid_y = torch.ones_like(euclid_dist_an)
            euclid_loss = self.ranking_loss(euclid_dist_an, euclid_dist_ap, euclid_y)
            loss += (1-alpha) * euclid_loss
        
        return self.loss_weight * loss
    
    def forward(self, feats, targets, feats_log_cov, alpha=0.1, **kwargs):
        if self.hard_mining:
            return self.hard_mining_triplet_loss_forward(feats, feats_log_cov, targets, alpha=alpha)
        else:
            raise NotImplementedError()
