# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcls.models.builder import HEADS
from mmcls.models.heads.base_head import BaseHead
from mmcls.models.losses import Accuracy
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.builder import build_loss
from mmcv.cnn.utils import kaiming_init

from .fc_module import FcModule


@HEADS.register_module()
class ProbabilisticReIDHead(BaseHead):
    """Probabilistic head for re-identification.

    Args:
        num_fcs (int): Number of fcs.
        in_channels (int): Number of channels in the input.
        fc_channels (int): Number of channels in the fcs.
        out_channels (int): Number of channels in the output.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to None.
        num_classes (int, optional): Number of the identities. Default to None.
        lce_sample_weight (float, optional): The weight of the ce loss from samples.
            Default to 0.1.
        num_samples (int, optional): Number of samples to calculate the ce loss.
            Default to 10.
        loss (dict, optional): Cross entropy loss to train the
            re-identificaiton module.
        loss_pairwise (dict, optional): Triplet loss to train the
            re-identificaiton module.
        topk (int, optional): Calculate topk accuracy. Default to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to dict(type='Normal',layer='Linear', mean=0, std=0.01,
            bias=0).
    """

    def __init__(self,
                 num_fcs,
                 in_channels,
                 fc_channels,
                 out_channels,
                 lce_sample_weight=0.1,
                 num_samples=10,
                 norm_cfg=None,
                 act_cfg=None,
                 num_classes=None,
                 loss=None,
                 loss_pairwise=None,
                 loss_uncertainty=None,
                 topk=(1, ),
                 init_cfg=dict(
                     type='Normal', layer='Linear', mean=0, std=0.01, bias=0)):
        super(ProbabilisticReIDHead, self).__init__(init_cfg)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        if not loss:
            if isinstance(num_classes, int):
                warnings.warn('Since cross entropy is not set, '
                              'the num_classes will be ignored.')
            if not loss_pairwise:
                raise ValueError('Please choose at least one loss in '
                                 'triplet loss and cross entropy loss.')
        elif not isinstance(num_classes, int):
            raise TypeError('The num_classes must be a current number, '
                            'if there is cross entropy loss.')
        self.loss_cls = build_loss(loss) if loss else None
        self.loss_triplet = build_loss(
            loss_pairwise) if loss_pairwise else None
        self.loss_uncertainty = build_loss(loss_uncertainty) if loss_uncertainty else None
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.lce_sample_weight = lce_sample_weight
        self.num_samples = num_samples
        self.accuracy = Accuracy(topk=self.topk)
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        """Initialize direct modeling and fc layers."""
        self.fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            in_channels = self.in_channels if i == 0 else self.fc_channels
            self.fcs.append(
                FcModule(in_channels, self.fc_channels, self.norm_cfg,
                         self.act_cfg))
        
        in_channels = self.in_channels if self.num_fcs == 0 else \
            self.fc_channels
            
        #? Create module for feature distribution mean explicit estimation
        self.feat_mean = FcModule(in_channels, self.out_channels)
        #? Create layer for feature covariance estimation (diagonal)
        self.feat_cov = FcModule(in_channels, self.out_channels)

        if self.loss_cls:
            self.bn = nn.BatchNorm1d(self.out_channels)
            self.classifier = nn.Linear(self.out_channels, self.num_classes)

    @auto_fp16()
    def forward_train(self, x):
        """Model forward."""
        for m in self.fcs:
            x = m(x)
        feats = self.feat_mean(x)
        feats_logcov = self.feat_cov(x)
        if self.loss_cls:
            feats_bn = self.bn(feats)
            cls_score = self.classifier(feats_bn)
            
            #? Run classifier on a number of samples
            feats_cov = torch.exp(feats_logcov)
            feat_dim = feats_cov.shape[-1]
            univ_norm_dists = torch.distributions.MultivariateNormal(torch.zeros(feat_dim), torch.eye(feat_dim))
            standard_norm = (univ_norm_dists.rsample((feats.shape[0]*self.num_samples,)).reshape(
                                feats.shape[0], self.num_samples, feat_dim)).to(feats_cov.device)
            feat_samples = torch.unsqueeze(feats, 1) + torch.mul(standard_norm, torch.unsqueeze(feats_cov, 1))
            
            cls_sample_score = self.classifier(self.bn(feat_samples.reshape(-1, feat_dim)))
            
            return (feats, feats_logcov, feats_cov, cls_score, cls_sample_score)
        return (feats, feats_logcov, feats_cov)

    @force_fp32(apply_to=('feats', 'feats_logcov', 'feats_cov', 'cls_score', 'cls_sample_score'))
    def loss(self, gt_label, feats, feats_logcov, feats_cov, cls_score=None, cls_sample_score=None):
        """Compute losses."""
        losses = dict()

        if self.loss_triplet:
            losses['triplet_loss'] = self.loss_triplet(feats, gt_label, cov=feats_cov)

        if self.loss_cls:
            assert cls_score is not None
            assert cls_sample_score is not None
            assert self.loss_uncertainty is not None, 'Uncertainty loss must be specified for classification'

            ce_loss = self.loss_cls(cls_score, gt_label)
            gt_label_repeat = torch.repeat_interleave(gt_label, self.num_samples)
            ce_loss_sample = self.loss_cls(cls_sample_score, gt_label_repeat)
            
            losses['ce_loss'] = (1 - self.lce_sample_weight) * ce_loss + self.lce_sample_weight * ce_loss_sample
            losses['uncertainty_loss'] = self.loss_uncertainty(feats_logcov)
            
            # compute accuracy
            acc = self.accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }

        return losses
